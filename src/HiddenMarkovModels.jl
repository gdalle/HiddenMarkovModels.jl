"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.
"""
module HiddenMarkovModels

using Base: RefValue
using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity, logdensityof
using DocStringExtensions
using FillArrays: Fill
using LinearAlgebra: Diagonal, axpy!, dot, ldiv!, lmul!, mul!
using PrecompileTools: @compile_workload
using Random: Random, AbstractRNG, default_rng
using Requires: @require
using SimpleUnPack: @unpack
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, nnz, nonzeros, nzrange
using StatsAPI: StatsAPI, fit, fit!

export AbstractHMM, HMM
export initialization, transition_matrix, obs_distributions
export fit!, logdensityof
export viterbi, forward, forward_backward, baum_welch
export seq_limits

const DESCRIBE_CONTROL_STARTS = """
- `control_seq`: a control sequence with the same length as `obs_seq`
- `seq_ends`: the indices at which each subsequence inside `obs_seq` and `control_seq` finishes, useful in the case of multiple sequences
"""

include("types/abstract_hmm.jl")

include("utils/linalg.jl")
include("utils/check.jl")
include("utils/probvec_transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")
include("utils/lightcategorical.jl")
include("utils/limits.jl")

include("inference/forward.jl")
include("inference/viterbi.jl")
include("inference/forward_backward.jl")
include("inference/baum_welch.jl")
include("inference/logdensity.jl")

include("types/hmm.jl")

function test_equal_hmms end
function test_coherent_algorithms end

if !isdefined(Base, :get_extension)
    function __init__()
        @require Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f" begin
            include("../ext/HiddenMarkovModelsDistributionsExt.jl")
        end
        @require HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7" begin
            include("../ext/HiddenMarkovModelsHMMBaseExt.jl")
        end
        @require Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40" begin
            include("../ext/HiddenMarkovModelsTestExt.jl")
        end
    end
end

# include("precompile.jl")

end
