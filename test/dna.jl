using DensityInterface
using HiddenMarkovModels
using Random: AbstractRNG
using RequiredInterfaces: check_interface_implemented
using SimpleUnPack
using StatsAPI
using Test

struct Dirac{T}
    val::T
end

Base.rand(::AbstractRNG, d::Dirac) = d.val
DensityInterface.DensityKind(::Dirac) = HasDensity()
DensityInterface.logdensityof(d::Dirac, x) = x == d.val ? 0.0 : -Inf

"""
    DNACodingHMM <: AbstractHMM

Custom implementation of an autoregressive HMM based on a standard HMM.

This describes the behavior of DNA as it moves from coding to noncoding segments.
In theory, the state is a character `coding` and the observation is a character `nucleotide`.
In practice, the state is a character couple `(coding, nucleotide)` and the observation is the exact same `nucleotide`.

# Notations

Coding:

| 1  | 2  |
|----|----|
| C  | N  |

Emissions:

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| A | T | G | C |

States:

| 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|-------|-------|-------|-------|-------|-------|-------|-------|
| (C,A) | (C,T) | (C,G) | (C,C) | (N,A) | (N,T) | (N,G) | (N,C) |

# Fields

- `cod_init::Vector{Float64}`: initial coding distribution
- `nuc_init::Vector{Float64}`: initial nucleotide distribution
- `cod_trans::Matrix{Float64}`: transition matrix between coding and noncoding
- `nuc_trans::Array{Float64,2}`: pair of transition matrices between nucleotides in coding and noncoding state
"""
struct DNACodingHMM <: AbstractHMM
    cod_init::Vector{Float64}
    nuc_init::Vector{Float64}
    cod_trans::Matrix{Float64}
    nuc_trans::Array{Float64,3}
    function DNACodingHMM(; cod_init, nuc_init, cod_trans, nuc_trans)
        @assert length(cod_init) == 2
        @assert length(nuc_init) == 4
        @assert size(cod_trans) == (2, 2)
        @assert size(nuc_trans) == (2, 4, 4)
        return new(cod_init, nuc_init, cod_trans, nuc_trans)
    end
end

get_coding(state) = 1 + (state - 1) ÷ 4
get_nucleotide(state) = 1 + (state - 1) % 4
get_state(coding, nucleotide) = 4(coding - 1) + nucleotide

@test get_coding.(1:8) == repeat(1:2; inner=4)
@test get_nucleotide.(1:8) == repeat(1:4; outer=2)
@test get_state.(get_coding.(1:8), get_nucleotide.(1:8)) == collect(1:8)

Base.length(dchmm::DNACodingHMM) = 8

function HMMs.initial_distribution(dchmm::DNACodingHMM)
    return repeat(dchmm.cod_init; inner=4) .* repeat(dchmm.nuc_init; outer=2)
end

function HMMs.transition_matrix(dchmm::DNACodingHMM)
    @unpack cod_trans, nuc_trans = dchmm
    A = Matrix{Float64}(undef, 8, 8)
    for c1 in 1:2, n1 in 1:4, c2 in 1:2, n2 in 1:4
        s1, s2 = get_state(c1, n1), get_state(c2, n2)
        A[s1, s2] = cod_trans[c1, c2] * nuc_trans[c1, n1, n2]
    end
    return A
end

function HMMs.obs_distribution(::DNACodingHMM, s::Integer)
    return Dirac(get_nucleotide(s))
end

function StatsAPI.fit!(
    dchmm::DNACodingHMM, init_count, trans_count, obs_seq, state_marginals
)
    # Initializations
    for c in 1:2
        dchmm.cod_init[c] = sum(init_count[get_state(c, n)] for n in 1:4)
    end
    for n in 1:4
        dchmm.nuc_init[n] = sum(init_count[get_state(c, n)] for c in 1:2)
    end
    HMMs.sum_to_one!(dchmm.cod_init)
    HMMs.sum_to_one!(dchmm.nuc_init)

    # Transitions
    for c1 in 1:2, c2 in 1:2
        dchmm.cod_trans[c1, c2] = sum(
            trans_count[get_state(c1, n1), get_state(c2, n2)] for n1 in 1:4, n2 in 1:4
        )
    end
    for c1 in 1:2, n1 in 1:4, n2 in 1:4
        dchmm.nuc_trans[c1, n1, n2] = sum(
            trans_count[get_state(c1, n1), get_state(c2, n2)] for c2 in 1:2
        )
    end
    foreach(HMMs.sum_to_one!, eachrow(dchmm.cod_trans))
    foreach(HMMs.sum_to_one!, eachrow(@view dchmm.nuc_trans[1, :, :]))
    foreach(HMMs.sum_to_one!, eachrow(@view dchmm.nuc_trans[2, :, :]))

    return nothing
end

@test check_interface_implemented(AbstractHMM, DNACodingHMM)

dchmm = DNACodingHMM(;
    cod_init=rand_prob_vec(2),
    nuc_init=rand_prob_vec(4),
    cod_trans=rand_trans_mat(2),
    nuc_trans=permutedims(cat(rand_trans_mat(4), rand_trans_mat(4); dims=3), (3, 1, 2)),
);

@unpack state_seq, obs_seq = rand(dchmm, 10_000);

most_likely_coding_seq = get_coding.(viterbi(dchmm, obs_seq));

mc = MarkovChain(ones(4) ./ 4, rand_trans_mat(4))
fit!(mc, obs_seq)

dchmm_init = DNACodingHMM(;
    cod_init=rand(2),
    nuc_init=rand(4),
    cod_trans=rand_trans_mat(2),
    # using transition_matrix(mc) as initialization below seems to be worse
    nuc_trans=permutedims(cat(rand_trans_mat(4), rand_trans_mat(4); dims=3), (3, 1, 2)),
);

dchmm_est, logL_evolution = baum_welch(dchmm_init, obs_seq; rtol=1e-7, max_iterations=100);

logL_evolution

sum(abs, dchmm_init.cod_trans - dchmm.cod_trans) / (2 * 2)
sum(abs, dchmm_est.cod_trans - dchmm.cod_trans) / (2 * 2)

sum(abs, dchmm_init.nuc_trans - dchmm.nuc_trans) / (2 * 4 * 4)
sum(abs, dchmm_est.nuc_trans - dchmm.nuc_trans) / (2 * 4 * 4)
