function benchmarkables_hiddenmarkovmodels(rng::AbstractRNG; configuration, algos)
    (; sparse, custom_dist, nb_states, obs_dim, seq_length, nb_seqs, bw_iter) =
        configuration

    # Model
    init = ones(nb_states) / nb_states
    if sparse
        trans = spdiagm(
            0 => ones(nb_states) / 2,
            +1 => ones(nb_states - 1) / 2,
            -(nb_states - 1) => ones(1) / 2,
        )
    else
        trans = ones(nb_states, nb_states) / nb_states
    end

    if custom_dist
        dists = [LightDiagNormal(i .* ones(obs_dim), ones(obs_dim)) for i in 1:nb_states]
    else
        if obs_dim == 1
            dists = [Normal(i, 1.0) for i in 1:nb_states]
        else
            dists = [
                MvNormal(i .* ones(obs_dim), Diagonal(ones(obs_dim))) for i in 1:nb_states
            ]
        end
    end
    hmm = HiddenMarkovModels.HMM(init, trans, dists)

    # Data
    obs_seqs = [rand(rng, hmm, seq_length).obs_seq for _ in 1:nb_seqs]
    obs_seq = reduce(vcat, obs_seqs)
    control_seq = fill(nothing, length(obs_seq))
    seq_ends = cumsum(length.(obs_seqs))

    # Benchmarks
    benchs = Dict()

    if "rand" in algos
        benchs["rand"] = @benchmarkable begin
            [rand($hmm, $seq_length).obs_seq for _ in 1:($nb_seqs)]
        end evals = 1 samples = 100
    end

    if "logdensity" in algos
        benchs["logdensity"] = @benchmarkable begin
            logdensityof($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100
    end

    if "forward" in algos
        benchs["forward"] = @benchmarkable begin
            forward($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100
        benchs["forward!"] = @benchmarkable begin
            forward!(f_storage, $hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100 setup = (
            f_storage = initialize_forward($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        )
    end

    if "viterbi" in algos
        benchs["viterbi"] = @benchmarkable begin
            viterbi($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100
        benchs["viterbi!"] = @benchmarkable begin
            viterbi!(v_storage, $hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100 setup = (
            v_storage = initialize_viterbi($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        )
    end

    if "forward_backward" in algos
        benchs["forward_backward"] = @benchmarkable begin
            forward_backward($hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100
        benchs["forward_backward!"] = @benchmarkable begin
            forward_backward!(fb_storage, $hmm, $obs_seq, $control_seq; seq_ends=$seq_ends)
        end evals = 1 samples = 100 setup = (
            fb_storage = initialize_forward_backward(
                $hmm, $obs_seq, $control_seq; seq_ends=$seq_ends
            )
        )
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            baum_welch(
                $hmm,
                $obs_seq,
                $control_seq;
                seq_ends=$seq_ends,
                max_iterations=$bw_iter,
                atol=-Inf,
                loglikelihood_increasing=false,
            )
        end evals = 1 samples = 100
        benchs["baum_welch!"] = @benchmarkable begin
            baum_welch!(
                fb_storage,
                logL_evolution,
                $hmm,
                $obs_seq,
                $control_seq;
                seq_ends=$seq_ends,
                max_iterations=$bw_iter,
                atol=-Inf,
                loglikelihood_increasing=false,
            )
        end evals = 1 samples = 100 setup = (
            fb_storage = initialize_forward_backward(
                $hmm, $obs_seq, $control_seq; seq_ends=$seq_ends
            );
            logL_evolution = Float64[];
            sizehint!(logL_evolution, $bw_iter)
        )
    end

    return benchs
end
