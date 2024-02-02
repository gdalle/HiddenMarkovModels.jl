struct pomegranateImplem <: Implementation end

function HMMBenchmark.build_model(
    rng::AbstractRNG, implem::pomegranateImplem; instance::Instance
)
    np = pyimport("numpy")
    torch = pyimport("torch")
    torch.set_default_dtype(torch.float64)
    pomegranate_distributions = pyimport("pomegranate.distributions")
    pomegranate_hmm = pyimport("pomegranate.hmm")

    (; nb_states, bw_iter) = instance
    (; init, trans, means, stds) = build_params(rng; instance)

    starts = torch.tensor(np.array(init))
    ends = torch.ones(nb_states) * 1e-10
    edges = torch.tensor(np.array(trans))

    distributions = pylist([
        pomegranate_distributions.Normal(;
            means=torch.tensor(np.array(means[:, i])),
            covs=torch.square(torch.tensor(np.array(stds[:, i] .^ 2))),
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
    rng::AbstractRNG, implem::pomegranateImplem; instance::Instance, algos::Vector{String}
)
    np = pyimport("numpy")
    torch = pyimport("torch")
    torch.set_default_dtype(torch.float64)
    (; obs_dim, seq_length, nb_seqs) = instance

    hmm = build_model(rng, implem; instance)
    data = randn(rng, nb_seqs, seq_length, obs_dim)

    obs_tens_py = torch.tensor(np.array(data))

    benchs = Dict()

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            $(hmm.forward)($obs_tens_py)
        end evals = 1 samples = 100
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            $(hmm.forward)($obs_tens_py)
        end evals = 1 samples = 100
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            $(hmm.forward_backward)($obs_tens_py)
        end evals = 1 samples = 100
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            hmm_guess.fit($obs_tens_py)
        end evals = 1 samples = 100 setup = (
            hmm_guess = build_model($rng, $implem; instance=$instance)
        )
    end

    return benchs
end
