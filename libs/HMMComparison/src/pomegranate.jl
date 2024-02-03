struct pomegranateImplem <: Implementation end
Base.string(::pomegranateImplem) = "pomegranate"

function HMMBenchmark.build_model(
    implem::pomegranateImplem, instance::Instance, params::Params
)
    torch = pyimport("torch")
    pomegranate_distributions = pyimport("pomegranate.distributions")
    pomegranate_hmm = pyimport("pomegranate.hmm")
    torch.set_default_dtype(torch.float64)

    (; nb_states, bw_iter) = instance
    (; init, trans, means, stds) = params

    starts = torch.tensor(Py(init).to_numpy())
    ends = torch.ones(nb_states) * 1e-10
    edges = torch.tensor(Py(trans).to_numpy())

    distributions = pylist([
        pomegranate_distributions.Normal(;
            means=torch.tensor(Py(means[:, i]).to_numpy()),
            covs=torch.square(torch.tensor(Py(stds[:, i] .^ 2).to_numpy())),
            covariance_type="diag",
        ) for i in 1:nb_states
    ])

    hmm = pomegranate_hmm.DenseHMM(;
        distributions=distributions,
        edges=edges,
        starts=starts,
        ends=ends,
        max_iter=bw_iter,
        tol=1e-10,
        verbose=false,
    )

    return hmm
end

function HMMBenchmark.build_benchmarkables(
    implem::pomegranateImplem,
    instance::Instance,
    params::Params,
    data::AbstractArray{<:Real,3},
    algos::Vector{String},
)
    torch = pyimport("torch")
    torch.set_default_dtype(torch.float64)

    (; obs_dim, seq_length, nb_seqs) = instance
    hmm = build_model(implem, instance, params)

    obs_tens_torch_py = torch.tensor(Py(data).to_numpy())

    benchs = Dict()

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            $(hmm.forward)($obs_tens_torch_py)
        end evals = 1 samples = 100
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            $(hmm.forward_backward)($obs_tens_torch_py)
        end evals = 1 samples = 100
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            hmm_guess.fit($obs_tens_torch_py)
        end evals = 1 samples = 100 setup = (
            hmm_guess = build_model($implem, $instance, $params)
        )
    end

    return benchs
end
