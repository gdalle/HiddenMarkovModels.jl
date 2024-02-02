struct HiddenMarkovModelsImplem <: Implementation end

function build_model(rng::AbstractRNG, ::HiddenMarkovModelsImplem; instance::Instance)
    (; custom_dist, nb_states, obs_dim) = instance
    (; init, trans, means, stds) = build_params(rng; instance)

    if custom_dist
        dists = [LightDiagNormal(means[:, i], stds[:, i]) for i in 1:nb_states]
    else
        if obs_dim == 1
            dists = [Normal(means[1, i], stds[1, i]) for i in 1:nb_states]
        else
            dists = [MvNormal(means[:, i], Diagonal(stds[:, i])) for i in 1:nb_states]
        end
    end

    hmm = HiddenMarkovModels.HMM(init, trans, dists)
    return hmm
end

function build_benchmarkables(
    rng::AbstractRNG,
    implem::HiddenMarkovModelsImplem;
    instance::Instance,
    algos::Vector{String},
)
    (; custom_dist, obs_dim, seq_length, nb_seqs, bw_iter) = instance

    hmm = build_model(rng, implem; instance)
    data = randn(rng, nb_seqs, seq_length, obs_dim)

    if obs_dim == 1
        obs_seqs = [[data[k, t, 1] for t in 1:seq_length] for k in 1:nb_seqs]
    else
        if custom_dist
            obs_seqs = [[data[k, t, :] for t in 1:seq_length] for k in 1:nb_seqs]
        else
            obs_seqs = [data[k, :, :] for k in 1:nb_seqs]
        end
    end
    obs_seq = reduce(vcat, obs_seqs)
    control_seq = Fill(nothing, duration(obs_seq))
    seq_ends = cumsum(length.(obs_seqs))

    benchs = Dict()

    if "rand" in algos
        benchs["rand"] = @benchmarkable begin
            [rand($rng, $hmm, $seq_length).obs_seq for _ in 1:($nb_seqs)]
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
                hmm_guess,
                $obs_seq,
                $control_seq;
                seq_ends=$seq_ends,
                max_iterations=$bw_iter,
                atol=-Inf,
                loglikelihood_increasing=false,
            )
        end evals = 1 samples = 100 setup = (
            hmm_guess = build_model($rng, $implem; instance=$instance);
            fb_storage = initialize_forward_backward(
                hmm_guess, $obs_seq, $control_seq; seq_ends=$seq_ends
            );
            logL_evolution = Float64[];
            sizehint!(logL_evolution, $bw_iter)
        )
    end

    return benchs
end
