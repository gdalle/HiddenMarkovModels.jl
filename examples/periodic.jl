# # Periodic HMM

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using Random
using SimpleUnPack
using StatsAPI
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

#-
struct PeriodicHMM{T<:Number,D,L} <: AbstractHMM
    init::Vector{T}
    trans_per::NTuple{L,Matrix{T}}
    dists_per::NTuple{L,Vector{D}}
end

#-

period(::PeriodicHMM{T,D,L}) where {T,D,L} = L

function HMMs.initialization(hmm::PeriodicHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::PeriodicHMM, t::Integer)
    return hmm.trans_per[(t - 1) % period(hmm) + 1]
end

function HMMs.obs_distributions(hmm::PeriodicHMM, t::Integer)
    return hmm.dists_per[(t - 1) % period(hmm) + 1]
end

#-

init = [0.4, 0.6]
trans_per = ([0.8 0.2; 0.2 0.8], [0.6 0.4; 0.4 0.6])
dists_per = ([Normal(-1), Normal(+1)], [Normal(-2), Normal(+2)])
hmm = PeriodicHMM(init, trans_per, dists_per)

#-

T = 100
control_seq = 1:T
state_seq, obs_seq = rand(rng, hmm, control_seq)
@test sum(abs, obs_seq[1:2:end]) < 0.8 * sum(abs, obs_seq[2:2:end])  #src

#-

viterbi(hmm, obs_seq; control_seq)
forward(hmm, obs_seq; control_seq)
logdensityof(hmm, obs_seq; control_seq)
forward_backward(hmm, obs_seq; control_seq);

#-

function StatsAPI.fit!(
    hmm::PeriodicHMM{T},
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    fb_storage::HMMs.ForwardBackwardStorage,
) where {T}
    @unpack γ, ξ = fb_storage
    L, N = period(hmm), length(hmm)

    hmm.init .= zero(T)
    for l in 1:L
        hmm.trans_per[l] .= zero(T)
    end
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
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
        @views for i in 1:N
            HMMs.fit_in_sequence!(hmm.dists_per[l], i, obs_seq[times_l], γ[i, times_l])
        end
    end

    for l in 1:L
        HMMs.check_hmm(hmm; control=l)
    end
    return nothing
end

#-

init_guess = [0.5, 0.5]
trans_per_guess = ([0.7 0.3; 0.3 0.7], [0.5 0.5; 0.5 0.5])
dists_per_guess = ([Normal(-0.7), Normal(+0.7)], [Normal(-1.5), Normal(+1.5)])
hmm_guess = PeriodicHMM(init_guess, trans_per_guess, dists_per_guess)

#-

control_seqs = [1:rand(rng, T:(2T)) for k in 1:100];
obs_seqs = [rand(rng, hmm, control_seq).obs_seq for control_seq in control_seqs];

hmm_est, logL_evolution = baum_welch(
    hmm_guess,
    reduce(vcat, obs_seqs);
    control_seq=reduce(vcat, control_seqs),
    seq_ends=cumsum(length.(obs_seqs)),
)
@test HMMs.similar_hmms(hmm_est, hmm; control_seq=1:2, atol=0.05)  #src
