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

function benchmarkables_hiddenmarkovmodels(; configuration, algos)
    @unpack seq_length, nb_seqs, bw_iter = configuration
    hmm = rand_gaussian_hmm(; configuration)
    obs_seqs = [rand(hmm, seq_length).obs_seq for _ in 1:nb_seqs]

    obs_seq = reduce(vcat, obs_seqs)
    seq_ends = cumsum(length.(obs_seqs))

    benchs = Dict()

    if "rand" in algos
        benchs["rand"] = @benchmarkable begin
            [rand($hmm, $seq_length).obs_seq for _ in 1:($nb_seqs)]
        end
    end

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            logdensityof($hmm, $obs_seq; seq_ends=$seq_ends)
        end
    end

    if "forward" in algos
        benchs["forward_init"] = @benchmarkable begin
            initialize_forward_backward($hmm, $obs_seq; seq_ends=$seq_ends)
        end
        benchs["forward!"] = @benchmarkable begin
            forward!(f_storage, $hmm, $obs_seq; seq_ends=$seq_ends)
        end setup = (f_storage = initialize_forward($hmm, $obs_seq; seq_ends=$seq_ends))
    end

    if "viterbi" in algos
        benchs["viterbi_init"] = @benchmarkable begin
            initialize_viterbi($hmm, $obs_seq; seq_ends=$seq_ends)
        end
        benchs["viterbi!"] = @benchmarkable begin
            viterbi!(v_storage, $hmm, $obs_seq; seq_ends=$seq_ends)
        end setup = (v_storage = initialize_viterbi($hmm, $obs_seq; seq_ends=$seq_ends))
    end

    if "forward_backward" in algos
        benchs["forward_backward_init"] = @benchmarkable begin
            initialize_forward_backward($hmm, $obs_seq; seq_ends=$seq_ends)
        end
        benchs["forward_backward!"] = @benchmarkable begin
            forward_backward!(fb_storage, $hmm, $obs_seq; seq_ends=$seq_ends)
        end setup = (
            fb_storage = initialize_forward_backward($hmm, $obs_seq; seq_ends=$seq_ends)
        )
    end

    if "baum_welch" in algos
        benchs["baum_welch!"] = @benchmarkable begin
            baum_welch!(
                fb_storage,
                logL_evolution,
                $hmm,
                $obs_seq;
                seq_ends=$seq_ends,
                max_iterations=$bw_iter,
                atol=-Inf,
                loglikelihood_increasing=false,
            )
        end setup = (
            fb_storage = initialize_forward_backward($hmm, $obs_seq; seq_ends=$seq_ends);
            logL_evolution = Float64[];
            sizehint!(logL_evolution, $bw_iter)
        )
    end

    return benchs
end
