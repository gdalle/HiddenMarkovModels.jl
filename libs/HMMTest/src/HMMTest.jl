module HMMTest

using BenchmarkTools: @ballocated
using HiddenMarkovModels
using HiddenMarkovModels: AbstractVectorOrNTuple
import HiddenMarkovModels as HMMs
using HMMBase: HMMBase
using JET: @test_opt, @test_call
using Random: AbstractRNG
using Statistics: mean
using Test: @test, @testset, @test_broken

export transpose_hmm
export test_equal_hmms, test_coherent_algorithms
export test_identical_hmmbase
export test_allocations
export test_type_stability

include("utils.jl")
include("coherence.jl")
include("allocations.jl")
include("hmmbase.jl")
include("jet.jl")

end
