function rand_gaussian_hmm(; configuration)
    @unpack sparse, nb_states, obs_dim = configuration
    init = ones(nb_states)
    trans = if !sparse
        rand_trans_mat(nb_states)
    else
        aux = SymTridiagonal(rand(nb_states), rand(nb_states - 1))
        aux ./ sum(aux; dims=2)
    end
    if obs_dim == 1
        dists = [Normal(randn(), 1.0) for _ in 1:nb_states]
    else
        dists = [LightDiagNormal(randn(obs_dim), ones(obs_dim)) for _ in 1:nb_states]
    end
    hmm = HiddenMarkovModels.HMM(init, trans, dists)
    return hmm
end

function rand_seqs(; configuration)
    @unpack obs_dim, seq_length, nb_seqs = configuration
    obs_seqs = [[randn(obs_dim) for _ in 1:seq_length] for _ in 1:nb_seqs]
    return MultiSeq(obs_seqs)
end

function benchmarkables_hiddenmarkovmodels(; configuration, algos)
    @unpack seq_length, nb_seqs, bw_iter = configuration
    hmm = rand_gaussian_hmm(; configuration)
    obs_seqs = MultiSeq([rand(hmm, seq_length).obs_seq for _ in 1:nb_seqs])

    benchs = Dict()

    if "rand" in algos
        benchs["rand"] = @benchmarkable begin
            [rand($hmm, $seq_length).obs_seq for _ in 1:($nb_seqs)]
        end
    end

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            logdensityof($hmm, $obs_seqs)
        end
    end

    if "forward" in algos
        benchs["forward_init"] = @benchmarkable begin
            initialize_forward_backward($hmm, $obs_seqs)
        end
        benchs["forward!"] = @benchmarkable begin
            forward!(f_storages, $hmm, $obs_seqs)
        end setup = (f_storages = initialize_forward($hmm, $obs_seqs))
    end

    if "viterbi" in algos
        benchs["viterbi_init"] = @benchmarkable begin
            initialize_viterbi($hmm, $obs_seqs)
        end
        benchs["viterbi!"] = @benchmarkable begin
            viterbi!(v_storages, $hmm, $obs_seqs)
        end setup = (v_storages = initialize_viterbi($hmm, $obs_seqs))
    end

    if "forward_backward" in algos
        benchs["forward_backward_init"] = @benchmarkable begin
            initialize_forward_backward($hmm, $obs_seqs)
        end
        benchs["forward_backward!"] = @benchmarkable begin
            forward_backward!(fb_storages, $hmm, $obs_seqs)
        end setup = (fb_storages = initialize_forward_backward($hmm, $obs_seqs))
    end

    if "baum_welch" in algos
        benchs["baum_welch_init"] = @benchmarkable begin
            initialize_baum_welch($hmm, $obs_seqs)
        end
        benchs["baum_welch!"] = @benchmarkable begin
            baum_welch!(
                bw_storage,
                $hmm,
                $obs_seqs;
                max_iterations=$bw_iter,
                atol=-Inf,
                loglikelihood_increasing=false,
            )
        end setup = (
            bw_storage = initialize_baum_welch($hmm, $obs_seqs; max_iterations=$bw_iter)
        )
    end

    return benchs
end