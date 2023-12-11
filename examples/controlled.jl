# # Control dependency

#=
Here, we give a example of controlled HMM (also called input-output HMM), in the special case of Markov switching regression.
=#

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
Random.seed!(rng, 63);

# ## Model

#=
A Markov switching regression is like a classical regression, except that the weights depend on the unobserved state of an HMM.
We can represent it with the following subtype of `AbstractHMM`, which has one vector of coefficients $\beta_i$ per state.
=#

struct ControlledGaussianHMM{T} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}
end

#=
Assuming we are in state $i$ with a vector of controls $u$, our observation is given by the linear model $y \sim \mathcal{N}(\beta_i^\top u, 1)$. 
=#

function HMMs.initialization(hmm::ControlledGaussianHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ControlledGaussianHMM)
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ControlledGaussianHMM, control::AbstractVector)
    return [Normal(dot(hmm.dist_coeffs[i], control), 1.0) for i in 1:length(hmm)]
end

#=
In this case, the transition matrix does not depend on the control.
=#

# ## Simulation

d = 3
init = [0.8, 0.2]
trans = [0.7 0.3; 0.3 0.7]
dist_coeffs = [-ones(d), ones(d)]
hmm = ControlledGaussianHMM(init, trans, dist_coeffs);

#=
Simulation requires a vector of controls, each being a vector itself with the right dimension.

Let us build several sequences of variable lengths.
=#

control_seqs = [[randn(rng, d) for t in 1:rand(100:200)] for k in 1:100];
obs_seqs = [rand(rng, hmm, control_seq).obs_seq for control_seq in control_seqs];

obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs));

# ## Inference

#=
Not much changes from the case with simple time dependency.
=#

best_state_seq, _ = viterbi(hmm, obs_seq; control_seq, seq_ends)

# ## Learning

#=
Once more, we override the `fit!` function.
The state-related parameters are estimated in the standard way.
Meanwhile, the observation coefficients are given by the formula for [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares).
=#

function StatsAPI.fit!(
    hmm::ControlledGaussianHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    control_seq::AbstractVector,
    seq_ends::AbstractVector{Int},
) where {T}
    @unpack γ, ξ = fb_storage
    N = length(hmm)

    hmm.init .= 0
    hmm.trans .= 0
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        @views hmm.trans .+= sum(ξ[t1:t2])
    end
    hmm.init ./= sum(hmm.init)
    for row in eachrow(hmm.trans)
        row ./= sum(row)
    end

    U = reduce(hcat, control_seq)'
    y = obs_seq
    for i in 1:N
        W = sqrt.(Diagonal(γ[i, :]))
        hmm.dist_coeffs[i] = (W * U) \ (W * y)
    end
end

#=
Now we put it to the test.
=#

init_guess = [0.7, 0.3]
trans_guess = [0.6 0.4; 0.4 0.6]
dist_coeffs_guess = [-0.5 * ones(d), 0.5 * ones(d)]
hmm_guess = ControlledGaussianHMM(init_guess, trans_guess, dist_coeffs_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq; control_seq, seq_ends)
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
How did we perform?
=#

cat(transition_matrix(hmm_est), transition_matrix(hmm); dims=3)

#-

hcat(hmm_est.dist_coeffs[1], hmm.dist_coeffs[1])

#-

hcat(hmm_est.dist_coeffs[2], hmm.dist_coeffs[2])

# ## Tests  #src

HMMs.test_coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)  #src
