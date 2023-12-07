using Distributions
using HMMBase: HMMBase
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels:
    LightDiagNormal, LightCategorical, similar_hmms, coherent_algorithms
using LinearAlgebra
using Random: Random, AbstractRNG, default_rng, seed!
using SimpleUnPack
using SparseArrays
using Test

rng = default_rng()
seed!(rng, 63)

function coherent_hmmbase(
    rng::AbstractRNG, hmm::AbstractHMM, hmm_guess::AbstractHMM; T::Integer
)
    sim = rand(rng, hmm, T)
    obs_mat = collect(reduce(hcat, sim.obs_seq)')

    obs_seq = vcat(sim.obs_seq, sim.obs_seq)
    seq_ends = [length(sim.obs_seq), 2 * length(sim.obs_seq)]

    hmm_base = HMMBase.HMM(deepcopy(hmm))
    hmm_guess_base = HMMBase.HMM(deepcopy(hmm_guess))

    logL_base = HMMBase.forward(hmm_base, obs_mat)[2]
    logL = logdensityof(hmm, obs_seq; seq_ends)
    if !(logL ≈ 2logL_base)
        @warn "Logdensity incoherent with HMMBase"
        return false
    end

    α_base, logL_forward_base = HMMBase.forward(hmm_base, obs_mat)
    α, logL_forward = forward(hmm, obs_seq; seq_ends)
    if !isapprox(α[:, 1:T], α_base') || !isapprox(α[:, (T + 1):(2T)], α_base')
        @warn "Forward filtered marginals incoherent with HMMBase"
        return false
    elseif !(logL_forward ≈ 2logL_forward_base)
        @warn "Forward loglikelihood incoherent with HMMBase"
        return false
    end

    q_base = HMMBase.viterbi(hmm_base, obs_mat)
    q, logL_viterbi = viterbi(hmm, obs_seq; seq_ends)
    # Viterbi decoding can vary in case of (infrequent) ties
    if mean(q[1:T] .== q_base) < 0.9 || mean(q[(T + 1):(2T)] .== q_base) < 0.9
        @warn "Viterbi state sequence incoherent with HMMBase"
        return false
    end

    γ_base = HMMBase.posteriors(hmm_base, obs_mat)
    γ, logL_forward_backward = forward_backward(hmm, obs_seq; seq_ends)
    if !isapprox(γ[:, 1:T], γ_base') || !isapprox(γ[:, (T + 1):(2T)], γ_base')
        @warn "Forward-backward filtered marginals incoherent with HMMBase"
        return false
    end

    hmm_est_base, hist_base = HMMBase.fit_mle(hmm_guess_base, obs_mat; maxiter=10, tol=-Inf)
    logL_evolution_base = hist_base.logtots
    hmm_est, logL_evolution = baum_welch(
        hmm_guess, obs_seq; seq_ends, max_iterations=10, atol=-Inf
    )
    if !isapprox(logL_evolution[(begin + 1):end], 2 * logL_evolution_base[begin:(end - 1)])
        @warn "Baum-Welch loglikelihood evolution incoherent with HMMBase"
        return false
    elseif !similar_hmms(hmm_est, HMM(hmm_est_base); atol=1e-5, test_init=true)
        @warn "Baum-Welch estimate incoherent with HMMBase"
        return false
    end

    return true
end

## Distributions

@testset "Categorical" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [Categorical([0.2, 0.8]), Categorical([0.8, 0.2])]
    dists_guess = [Categorical([0.3, 0.7]), Categorical([0.7, 0.3])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 200, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_hmmbase(rng, hmm, hmm_guess; T)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

@testset "Normal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [Normal(-1), Normal(+1)]
    dists_guess = [Normal(-0.5), Normal(+0.5)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 100, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_hmmbase(rng, hmm, hmm_guess; T)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

@testset "DiagNormal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    D = 3
    dists = [MvNormal(-ones(D), I), MvNormal(+ones(D), I)]
    dists_guess = [MvNormal(-ones(D) / 2, I), MvNormal(+ones(D) / 2, I)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 100, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_hmmbase(rng, hmm, hmm_guess; T)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

## Light distributions

@testset "LightCategorical" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    dists = [LightCategorical([0.2, 0.8]), LightCategorical([0.8, 0.2])]
    dists_guess = [LightCategorical([0.3, 0.7]), LightCategorical([0.7, 0.3])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 200, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

@testset "LightDiagNormal" begin
    init = [0.4, 0.6]
    init_guess = [0.5, 0.5]

    trans = [0.8 0.2; 0.2 0.8]
    trans_guess = [0.7 0.3; 0.3 0.7]

    D = 3
    dists = [LightDiagNormal(-ones(D), ones(D)), LightDiagNormal(+ones(D), ones(D))]
    dists_guess = [
        LightDiagNormal(-ones(D) / 2, ones(D)), LightDiagNormal(+ones(D) / 2, ones(D))
    ]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 100, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

## Weird arrays

@testset "Normal sparse" begin
    init = [0.2, 0.6, 0.2]
    trans = sparse([
        0.8 0.2 0.0
        0.0 0.8 0.2
        0.2 0.0 0.8
    ])
    dists = [Normal(-2), Normal(0), Normal(+2)]
    hmm = HMM(init, trans, dists)

    init_guess = [0.3, 0.4, 0.4]
    trans_guess = sparse([
        0.6 0.4 0.0
        0.0 0.6 0.4
        0.4 0.0 0.6
    ])
    dists_guess = [Normal(-1), Normal(0), Normal(+1)]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    T, K = 100, 100
    control_seq = fill(nothing, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_algorithms(rng, hmm, hmm_guess; control_seq, seq_ends, atol=0.05)
end

# Controlled

struct DiffusionHMM{R1,R2,R3} <: AbstractHMM
    init::Vector{R1}
    trans::Matrix{R2}
    means::Vector{R3}
end

HMMs.initialization(hmm::DiffusionHMM) = hmm.init

function HMMs.transition_matrix(hmm::DiffusionHMM, λ::Number)
    @assert 0 <= λ <= 1
    N = length(hmm)
    return (1 - λ) * hmm.trans + λ * ones(N, N) / N
end

function HMMs.obs_distributions(hmm::DiffusionHMM, λ::Number)
    return [Normal((1 - λ) * hmm.means[i]) for i in 1:length(hmm)]
end

@testset "Controlled" begin
    init = rand_prob_vec(rng, 2)
    trans = rand_trans_mat(rng, 2)
    means = randn(rng, 2)
    hmm = DiffusionHMM(init, trans, means)

    T, K = 100, 100
    control_seq = rand(rng, T * K)
    seq_ends = T:T:(T * K)
    @test coherent_algorithms(rng, hmm; control_seq, seq_ends, atol=0.05)
end
