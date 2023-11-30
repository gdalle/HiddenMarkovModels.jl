# # Periodic HMM

using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using Plots
using SimpleUnPack
using StatsAPI

# ## Structure

"""
    PeriodicHMM{L}

Basic implementation of a periodic HMM with time-dependent transition matrices and observation distributions, repeating every `L` time steps.
"""
struct PeriodicHMM{L,V<:AbstractVector,M<:AbstractMatrix,VD<:AbstractVector} <: AbstractHMM
    init::V
    trans_periodic::NTuple{L,M}
    dists_periodic::NTuple{L,VD}
end

period(::PeriodicHMM{L}) where {L} = L

Base.length(phmm::PeriodicHMM) = length(phmm.init)
HMMs.initialization(phmm::PeriodicHMM) = phmm.init

function HMMs.transition_matrix(phmm::PeriodicHMM, t::Integer)
    return phmm.trans_periodic[(t - 1) % period(hmm) + 1]
end

function HMMs.obs_distributions(phmm::PeriodicHMM, t::Integer)
    return phmm.dists_periodic[(t - 1) % period(hmm) + 1]
end

## Fitting

struct BaumWelchStoragePeriodicHMM <: HMMs.AbstractBaumWelchStorage end
function HMMs.initialize_baum_welch(::PeriodicHMM, fb_storages, obs_seqs)
    return BaumWelchStoragePeriodicHMM()
end

function fit_states!(hmm::PeriodicHMM, fb_storages::Vector{<:HMMs.ForwardBackwardStorage})
    L = period(hmm)
    # Reset
    hmm.init .= 0
    for l in 1:L
        hmm.trans_periodic[l] .= 0
    end
    # Accumulate sufficient stats
    for k in eachindex(fb_storages)
        @unpack γ, ξ = fb_storages[k]
        hmm.init .+= view(γ, :, 1)
        for t in eachindex(ξ)
            l = (t - 1) % L + 1
            hmm.trans_periodic[l] .+= ξ[t]
        end
    end
    # Normalize
    hmm.init ./= sum(hmm.init)
    for l in 1:L
        hmm.trans_periodic[l] ./= sum(hmm.trans_periodic[l]; dims=2)
    end
    return nothing
end

function fit_observations!(
    hmm::PeriodicHMM,
    fb_storages::Vector{<:HMMs.ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    for l in 1:L
        obs_seq_periodic = reduce(vcat, obs_seqs[k][l:L:end] for k in eachindex(obs_seqs))
        state_marginals_periodic = reduce(
            hcat, fb_storages[k].γ[:, l:L:end] for k in eachindex(fb_storages)
        )
        for i in 1:length(hmm)
            D = typeof(hmm.dists_periodic[l][i])
            x = obs_seq_periodic
            w = view(state_marginals_periodic, i, :)
            hmm.dists_periodic[l][i] = fit(D, x, w)
        end
    end
    return nothing
end

function StatsAPI.fit!(
    hmm::PeriodicHMM,
    ::BaumWelchStoragePeriodicHMM,
    fb_storages::Vector{<:HMMs.ForwardBackwardStorage},
    obs_seqs::Vector{<:Vector},
)
    fit_states!(hmm, fb_storages)
    fit_observations!(hmm, fb_storages, obs_seqs)
    return nothing
end

# ## Example

N = 2 # Number of hidden states
L = 10 # Period of the HMM
T = 50_000 # Number of observation

function make_trans(l, L)
    A = Matrix{Float64}(undef, 2, 2)
    A[1, 1] = 0.25 + 0.1 + 0.5cos(2π / L * l + 1)^2
    A[1, 2] = 0.25 - 0.1 + 0.5sin(2π / L * l + 1)^2
    A[2, 2] = 0.25 + 0.2 + 0.5cos(2π / L * (l - L / 3))^2
    A[2, 1] = 0.25 - 0.2 + 0.5sin(2π / L * (l - L / 3))^2
    return A
end

function make_dists(l, L, N)
    dists = [Normal(2i * cos(2π * l / L), i + cos(2π / L * (l - i / 2 + 1))^2) for i in 1:N]
    return dists
end

init = ones(N) / N;
trans_periodic = ntuple(l -> make_trans(l, L), L);
dists_periodic = ntuple(l -> make_dists(l, L, N), L);

hmm = PeriodicHMM(init, trans_periodic, dists_periodic);

state_seq, obs_seq = rand(hmm, T);

hmm_est, logL_evolution = baum_welch(hmm, obs_seq; max_iterations=100);
length(logL_evolution)

## Plotting

p = [plot(; xlabel="l", title="transitions from state $i") for i in 1:N]
for i in 1:N, j in 1:N
    plot!(
        p[i],
        1:L,
        [transition_matrix(hmm, l)[i, j] for l in 1:L];
        label="p$((i,j)) - true",
        c=j,
    )
    plot!(
        p[i],
        1:L,
        [transition_matrix(hmm_est, l)[i, j] for l in 1:L];
        label="p$((i,j)) - est",
        c=j,
        s=:dash,
    )
end
plot(p...; size=(1000, 500))

p = [plot(; xlabel="l", title="emissions from state $i") for i in 1:N]
for i in 1:N
    plot!(p[i], 1:L, [obs_distributions(hmm, l)[i].μ for l in 1:L]; label="μ - true", c=1)
    plot!(
        p[i],
        1:L,
        [obs_distributions(hmm_est, l)[i].μ for l in 1:L];
        label="μ - est",
        c=1,
        s=:dash,
    )
    plot!(p[i], 1:L, [obs_distributions(hmm, l)[i].σ for l in 1:L]; label="σ - true", c=2)
    plot!(
        p[i],
        1:L,
        [obs_distributions(hmm_est, l)[i].σ for l in 1:L];
        label="σ - est",
        c=2,
        s=:dash,
    )
end
plot(p...; size=(1000, 500))
