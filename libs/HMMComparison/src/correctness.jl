function compare_loglikelihoods(
    instance::Instance, params::Params, data::AbstractArray{<:Real,3}
)
    torch = pyimport("torch")
    jax = pyimport("jax")
    jnp = pyimport("jax.numpy")

    torch.set_default_dtype(torch.float64)
    jax.config.update("jax_enable_x64", true)

    (; obs_dim, seq_length, nb_seqs) = instance

    results = Dict{String,Any}()

    ## Data formats

    if obs_dim == 1
        obs_seqs = [[data[k, t, 1] for t in 1:seq_length] for k in 1:nb_seqs]
    else
        obs_seqs = [[data[k, t, :] for t in 1:seq_length] for k in 1:nb_seqs]
    end
    obs_seq = reduce(vcat, obs_seqs)
    control_seq = fill(nothing, length(obs_seq))
    seq_ends = cumsum(length.(obs_seqs))

    if obs_dim == 1
        obs_mat = reduce(vcat, data[k, :, 1] for k in 1:nb_seqs)
    else
        obs_mat = reduce(vcat, data[k, :, :] for k in 1:nb_seqs)
    end

    obs_mat_concat = reduce(vcat, data[k, :, :] for k in 1:nb_seqs)
    obs_mat_concat_py = Py(obs_mat_concat).to_numpy()
    obs_mat_len_py = Py(fill(seq_length, nb_seqs)).to_numpy()

    obs_tens_torch_py = torch.tensor(Py(data).to_numpy())
    obs_tens_jax_py = jnp.array(Py(data).to_numpy())

    # HiddenMarkovModels.jl

    implem1 = HiddenMarkovModelsImplem()
    hmm1 = build_model(implem1, instance, params)
    _, logLs1 = HiddenMarkovModels.forward(hmm1, obs_seq, control_seq; seq_ends)
    results[string(implem1)] = logLs1

    ## HMMBase.jl

    implem2 = HMMBaseImplem()
    hmm2 = build_model(implem2, instance, params)
    _, logL2 = HMMBase.forward(hmm2, obs_mat)
    results[string(implem2)] = logL2

    ## hmmlearn

    implem3 = hmmlearnImplem()
    hmm3 = build_model(implem3, instance, params)
    logL3 = hmm3.score(obs_mat_concat_py, obs_mat_len_py)
    results[string(implem3)] = pyconvert(Number, logL3)

    ## pomegranate

    implem4 = pomegranateImplem()
    hmm4 = build_model(implem4, instance, params)
    logαs4 = PyArray(hmm4.forward(obs_tens_torch_py))
    logLs4 = [logsumexp(logαs4[k, end, :]) for k in 1:nb_seqs]
    results[string(implem4)] = logLs4

    ## dynamax

    implem5 = dynamaxImplem()
    hmm5, dyn_params5 = build_model(implem5, instance, params)
    filter_vmap = jax.jit(jax.vmap(hmm5.filter; in_axes=pylist((pybuiltins.None, 0))))
    posterior5 = filter_vmap(dyn_params5, obs_tens_jax_py)
    logLs5 = PyArray(posterior5.marginal_loglik)
    results[string(implem5)] = logLs5

    return results
end
