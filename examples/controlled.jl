# # Controlled HMM

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using LinearAlgebra
using Random
using SimpleUnPack
using StatsAPI
using Test  #src

#-

rng = Random.default_rng()
Random.seed!(rng, 63)

# ## Model construction

struct ControlledGaussianHMM{T} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}
end

#-

function HMMs.initialization(hmm::ControlledGaussianHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ControlledGaussianHMM, control::AbstractVector)
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ControlledGaussianHMM, control::AbstractVector)
    return [Normal(dot(control, hmm.dist_coeffs[i])) for i in 1:length(hmm)]
end

#-

function StatsAPI.fit!(
    hmm::ControlledGaussianHMM{T},
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
    fb_storage::HMMs.ForwardBackwardStorage,
) where {T}
    @unpack γ, ξ = fb_storage
    N = length(hmm)

    hmm.init .= zero(T)
    hmm.trans .= zero(T)
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        @views hmm.trans .+= sum(ξ[t1:t2])
    end
    hmm.init ./= sum(hmm.init)
    for row in eachrow(hmm.trans)
        row ./= sum(row)
    end

    X = transpose(stack(control_seq))
    y = stack(obs_seq)
    for i in 1:N
        W = sqrt.(Diagonal(view(γ, i, :)))
        hmm.dist_coeffs[i] = (W * X) \ (W * y)
    end
end

# ## Demo

init = [0.4, 0.6]
trans = [0.7 0.3; 0.2 0.8]
dist_coeffs = [-ones(3), ones(3)]
hmm = ControlledGaussianHMM(init, trans, dist_coeffs);

#-

init_guess = [0.5, 0.5]
trans_guess = [0.5 0.5; 0.5 0.5]
dist_coeffs_guess = [-0.5 * ones(3), 0.5 * ones(3)]
hmm_guess = ControlledGaussianHMM(init_guess, trans_guess, dist_coeffs_guess);

#-

control_seqs = [[randn(rng, 3) for t in 1:rand(100:200)] for k in 1:100];
obs_seqs = [rand(rng, hmm, control_seq).obs_seq for control_seq in control_seqs];

obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs));

#-

hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq; control_seq, seq_ends)

# ## Tests  #src

test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)  #src
