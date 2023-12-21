function benchmarkables_pomegranate(rng::AbstractRNG; configuration, algos)
    torch = pyimport("torch")
    (; sparse, nb_states, obs_dim, seq_length, nb_seqs, bw_iter) = configuration

    # Model
    starts = torch.ones(nb_states) / nb_states
    edges = torch.ones(nb_states, nb_states) / nb_states
    distributions = pylist([
        pyimport("pomegranate.distributions").Normal(;
            means=i * torch.ones(obs_dim),
            covs=torch.square(torch.ones(obs_dim)),
            covariance_type="diag",
        ) for i in 1:nb_states
    ])
    hmm = pyimport("pomegranate.hmm").DenseHMM(;
        distributions=distributions,
        edges=edges,
        starts=starts,
        max_iter=bw_iter,
        tol=1e-10,
        verbose=false,
    )

    # Data
    obs_tens_py = torch.randn(nb_seqs, seq_length, obs_dim)

    # Benchmarks
    benchs = Dict()

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            pycall($(hmm.forward), $obs_tens_py)
        end
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            pycall($(hmm.forward), $obs_tens_py)
        end
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            pycall($(hmm.forward_backward), $obs_tens_py)
        end
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            pycall($(hmm.fit), $obs_tens_py)
        end
    end

    return benchs
end
