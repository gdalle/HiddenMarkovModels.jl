# # Time dependency

#=
Here, we demonstrate what to do transition and observation laws depend on the current time.
This time-dependent HMM is implemented as a particular case of controlled HMM.
=#

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMTest  #src
using Random
using StableRNGs
using StatsAPI
using Test  #src

#-

rng = StableRNG(63)

# ## Model

#=
We focus on the particular case of a periodic HMM with period `L`.
It has only one initialization vector, but `L` transition matrices and `L` vectors of observation distributions.
Once again we need to subtype `AbstractHMM`.
=#

struct PeriodicHMM{T<:Number,D,L} <: AbstractHMM
    init::Vector{T}
    trans_per::NTuple{L,Matrix{T}}
    dists_per::NTuple{L,Vector{D}}
end

#=
The interface definition is almost the same as in the homogeneous case, but we give the control variable (here the time) as an additional argument to `transition_matrix` and `obs_distributions`.
=#

period(::PeriodicHMM{T,D,L}) where {T,D,L} = L

function HMMs.initialization(hmm::PeriodicHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::PeriodicHMM, t::Integer)
    l = (t - 1) % period(hmm) + 1
    return hmm.trans_per[l]
end

function HMMs.obs_distributions(hmm::PeriodicHMM, t::Integer)
    l = (t - 1) % period(hmm) + 1
    return hmm.dists_per[l]
end

# ## Simulation

init = [0.8, 0.2]
trans_per = ([0.7 0.3; 0.3 0.7], [0.3 0.7; 0.7 0.3])
dists_per = ([Normal(-1.0), Normal(-2.0)], [Normal(+1.0), Normal(+2.0)])
hmm = PeriodicHMM(init, trans_per, dists_per);

#=
Since the behavior of the model depends on control variables, we need to pass these to the simulation routine (instead of just the number of time steps `T`).
=#

control_seq = 1:10
state_seq, obs_seq = rand(rng, hmm, control_seq);
obs_seq'
@test mean(obs_seq[1:2:end]) < 0 < mean(obs_seq[2:2:end])  #src

#=
The observations mostly alternate between positive and negative values, which is coherent with negative observation means at odd times and positive observation means at even times.

We now generate several sequences of variable lengths, for inference and learning tasks.
=#

control_seqs = [1:rand(100:200) for k in 1:1000]
obs_seqs = [rand(rng, hmm, control_seqs[k]).obs_seq for k in eachindex(control_seqs)];

obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs));

# ## Inference

#=
All three inference algorithms work in the same way, except that we need to provide the control sequence as a keyword argument.
=#

best_state_seq, _ = viterbi(hmm, obs_seq, control_seq; seq_ends)

#=
For Viterbi, unsurprisingly, the most likely state sequence aligns with the sign of the observations.
=#

vcat(obs_seq', best_state_seq')

# ## Learning

#=
When estimating parameters for a custom subtype of `AbstractHMM`, we have to override the fitting procedure after forward-backward (with an additional `control_seq` keyword argument).
The key is to split the observations according to which periodic parameter they belong to.
=#

function StatsAPI.fit!(
    hmm::PeriodicHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVector{Int},
) where {T}
    (; γ, ξ) = fb_storage
    L, N = period(hmm), length(hmm)

    hmm.init .= zero(T)
    for l in 1:L
        hmm.trans_per[l] .= zero(T)
    end
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        for l in 1:L
            hmm.trans_per[l] .+= sum(ξ[(t1 + l - 1):L:t2])
        end
    end
    hmm.init ./= sum(hmm.init)
    for l in 1:L, row in eachrow(hmm.trans_per[l])
        row ./= sum(row)
    end

    for l in 1:L
        times_l = Int[]
        for k in eachindex(seq_ends)
            t1, t2 = HMMs.seq_limits(seq_ends, k)
            append!(times_l, (t1 + l - 1):L:t2)
        end
        for i in 1:N
            HMMs.fit_in_sequence!(hmm.dists_per[l], i, obs_seq[times_l], γ[i, times_l])
        end
    end

    for l in 1:L
        HMMs.check_hmm(hmm; control=l)
    end
    return nothing
end

#=
Now let's test our procedure with a reasonable guess.
=#

init_guess = [0.7, 0.3]
trans_per_guess = ([0.6 0.4; 0.4 0.6], [0.4 0.6; 0.6 0.4])
dists_per_guess = ([Normal(-0.7), Normal(-1.7)], [Normal(+0.7), Normal(+1.7)])
hmm_guess = PeriodicHMM(init_guess, trans_per_guess, dists_per_guess);

#=
Naturally, Baum-Welch also requires knowing `control_seq`.
=#

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq, control_seq; seq_ends);
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
Did we do well?
=#

cat(transition_matrix(hmm_est, 1), transition_matrix(hmm, 1); dims=3)

#-

cat(transition_matrix(hmm_est, 2), transition_matrix(hmm, 2); dims=3)

#-

hcat(obs_distributions(hmm_est, 1), obs_distributions(hmm, 1))

#-

hcat(obs_distributions(hmm_est, 2), obs_distributions(hmm, 2))

# ## Tests  #src

test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.1, init=false)  #src
test_type_stability(rng, hmm, hmm_guess; control_seq, seq_ends)  #src
