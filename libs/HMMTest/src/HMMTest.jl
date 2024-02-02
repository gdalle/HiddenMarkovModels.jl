module HMMTest

using BenchmarkTools: @ballocated
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMBase: HMMBase
using JET: @test_opt, @test_call
using Random: AbstractRNG
using Statistics: mean
using Test: @test, @testset, @test_broken

export test_equal_hmms, test_coherent_algorithms
export test_identical_hmmbase
export test_allocations
export test_type_stability

include("coherence.jl")
include("allocations.jl")
include("hmmbase.jl")
include("jet.jl")

end
