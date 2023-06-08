var documenterSearchIndex = {"docs":
[{"location":"benchmarks/#Benchmarks","page":"Benchmarks","title":"Benchmarks","text":"","category":"section"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"These benchmarks were generated with the following setup:","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"using InteractiveUtils\nversioninfo()","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"The test case was a HMM with one-dimensional Gaussian observations, initialized randomly. Since HiddenMarkovModels.jl and HMMBase.jl give the exact same results, the only thing to compare is their speed of execution.","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"You can check out the complete benchmarking results in this JSON file created by BenchmarkTools.jl.","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"(Image: Logdensity benchmark)","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"(Image: Viterbi benchmark)","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"(Image: Forward-backward benchmark)","category":"page"},{"location":"benchmarks/","page":"Benchmarks","title":"Benchmarks","text":"(Image: Baum-Welch benchmark)","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [HiddenMarkovModels]","category":"page"},{"location":"api/#HiddenMarkovModels.HMMs","page":"API reference","title":"HiddenMarkovModels.HMMs","text":"HMMs\n\nAlias for the module HiddenMarkovModels.\n\n\n\n\n\n","category":"module"},{"location":"api/#HiddenMarkovModels.HiddenMarkovModels","page":"API reference","title":"HiddenMarkovModels.HiddenMarkovModels","text":"HiddenMarkovModels\n\nA Julia package for HMM modeling, simulation, inference and learning.\n\n\n\n\n\n","category":"module"},{"location":"api/#HiddenMarkovModels.HMM","page":"API reference","title":"HiddenMarkovModels.HMM","text":"HMM\n\nAlias for the struct HiddenMarkovModel.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.HiddenMarkovModel","page":"API reference","title":"HiddenMarkovModels.HiddenMarkovModel","text":"HiddenMarkovModel{SP<:StateProcess,OP<:ObservationProcess}\n\nCombination of a state and an observation process, amenable to simulation, inference and learning.\n\nFields\n\nstate_process::SP\nobs_process::OP\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.LogScale","page":"API reference","title":"HiddenMarkovModels.LogScale","text":"LogScale <: Scale\n\nTell algorithms to use full logarithmic scaling.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.NormalScale","page":"API reference","title":"HiddenMarkovModels.NormalScale","text":"NormalScale <: Scale\n\nTell algorithms to use no logarithmic scaling.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.ObservationProcess","page":"API reference","title":"HiddenMarkovModels.ObservationProcess","text":"ObservationProcess\n\nAbstract type for the observation part of an HMM.\n\nRequired methods\n\nBase.length(op)\ndistribution(op, i)\n\nOptional methods\n\nreestimate!(op, obs_seq, γ)\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.Scale","page":"API reference","title":"HiddenMarkovModels.Scale","text":"Scale\n\nAbstract type for dispatch-based choice of numerical robustness setting.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.SemiLogScale","page":"API reference","title":"HiddenMarkovModels.SemiLogScale","text":"SemiLogScale <: Scale\n\nTell algorithms to use partial logarithmic scaling.\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.StandardObservationProcess","page":"API reference","title":"HiddenMarkovModels.StandardObservationProcess","text":"StandardObservationProcess{D} <: ObservationProcess\n\nFields\n\ndistributions::AbstractVector{D}: one distribution per state\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.StandardStateProcess","page":"API reference","title":"HiddenMarkovModels.StandardStateProcess","text":"StandardStateProcess <: StateProcess\n\nFields\n\np::AbstractVector: initial distribution\nA::AbstractMatrix: transition matrix\n\n\n\n\n\n","category":"type"},{"location":"api/#HiddenMarkovModels.StateProcess","page":"API reference","title":"HiddenMarkovModels.StateProcess","text":"StateProcess\n\nAbstract type for the state part of an HMM.\n\nRequired methods\n\nBase.length(sp)\ninitial_distribution(sp)\ntransition_matrix(sp)\n\nOptional methods\n\nreestimate!(sp, p_count, A_count)\n\n\n\n\n\n","category":"type"},{"location":"api/#Base.rand-Tuple{HiddenMarkovModel, Integer}","page":"API reference","title":"Base.rand","text":"rand(hmm, T)\n\nSimulate an HMM for T time steps.\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{Random.AbstractRNG, HiddenMarkovModel, Integer}","page":"API reference","title":"Base.rand","text":"rand(rng, hmm, T)\n\nSimulate an HMM for T time steps with a specified rng.\n\n\n\n\n\n","category":"method"},{"location":"api/#DensityInterface.logdensityof-Tuple{HiddenMarkovModel, Any}","page":"API reference","title":"DensityInterface.logdensityof","text":"DensityInterface.logdensityof(hmm, obs_seq, scale=LogScale())\n\nApply the forward algorithm to compute the loglikelihood of a sequence of observations.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.baum_welch","page":"API reference","title":"HiddenMarkovModels.baum_welch","text":"baum_welch(hmm_init, obs_seqs, scale=LogScale(); max_iterations, rtol)\n\nApply the Baum-Welch algorithm to estimate the parameters of an HMM on multiple observation sequences, and return a tuple (hmm, logL_evolution).\n\n\n\n\n\n","category":"function"},{"location":"api/#HiddenMarkovModels.fit_from_sequence-Union{Tuple{D}, Tuple{Type{D}, AbstractVector, AbstractVector}} where D","page":"API reference","title":"HiddenMarkovModels.fit_from_sequence","text":"fit_from_sequence(::Type{D}, x, w)\n\nFit a distribution of type D based on a single sequence of observations x associated with a single sequence of weights w.\n\nDefault to StatsAPI.fit, with a special case for Distributions.jl and vectors of vectors (because this implementation of fit accepts matrices instead). Users are free to override this default for concrete distributions.\n\n\n\n\n\n","category":"method"},{"location":"api/#HiddenMarkovModels.forward_backward","page":"API reference","title":"HiddenMarkovModels.forward_backward","text":"forward_backward(hmm, obs_seq, scale=LogScale())\n\nApply the forward-backward algorithm to estimate the posterior state marginals of an HMM.\n\n\n\n\n\n","category":"function"},{"location":"api/#HiddenMarkovModels.viterbi-Tuple{HiddenMarkovModel, Any}","page":"API reference","title":"HiddenMarkovModels.viterbi","text":"viterbi(hmm, obs_seq, scale=LogScale())\n\nApply the Viterbi algorithm to compute the most likely sequence of states of an HMM.\n\n\n\n\n\n","category":"method"},{"location":"notations/#Notations","page":"Notations","title":"Notations","text":"","category":"section"},{"location":"notations/","page":"Notations","title":"Notations","text":"Our whole package is based on the following paper by Rabiner (1989):","category":"page"},{"location":"notations/","page":"Notations","title":"Notations","text":"A tutorial on hidden Markov models and selected applications in speech recognition","category":"page"},{"location":"notations/","page":"Notations","title":"Notations","text":"Please refer to it for mathematical explanations.","category":"page"},{"location":"notations/#State-process","page":"Notations","title":"State process","text":"","category":"section"},{"location":"notations/","page":"Notations","title":"Notations","text":"sp or state_process: a StateProcess\np: initial_distribution (vector of state probabilities)\nA: transition_matrix (matrix of transition probabilities)\nstate_seq: a sequence of states (vector of integers)","category":"page"},{"location":"notations/#Observation-process","page":"Notations","title":"Observation process","text":"","category":"section"},{"location":"notations/","page":"Notations","title":"Notations","text":"op or obs_process: an ObservationProcess\n(log)b: vector of observation (log)likelihoods by state for a single observation\n(log)B: matrix of observation (log)likelihoods by state for a sequence of observations\nobs_seq: a sequence of observations\nobs_seqs: several sequences of observations","category":"page"},{"location":"notations/#Forward-backward","page":"Notations","title":"Forward backward","text":"","category":"section"},{"location":"notations/","page":"Notations","title":"Notations","text":"α: forward variables\nc: forward variable inverse normalizations\nβ: backward variables\nγ: one-state marginals\nξ: two-state marginals\nlogL: loglikelihood of a sequence of observations","category":"page"},{"location":"#HiddenMarkovModels.jl","page":"Home","title":"HiddenMarkovModels.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A Julia package for HMM modeling, simulation, inference and learning.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This is an experimental package, the interface is not yet stable and the documentation is still insufficient. If you find something wrong or missing, please open an issue!","category":"page"},{"location":"#Main-features","page":"Home","title":"Main features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Efficiency\nallocation-free versions of core functions\nlinear algebra subroutines\nmultithreading\nGeneric state process\ndense or sparse transitions\nwith or without prior\nGeneric observation process\nDistributions.jl\nMeasureTheory.jl\nanything that follows DensityInterface.jl\nGeneric number types\nfloating point precision\nuncertainties\ndual numbers\nAutomatic differentiation of parameters\nin forward mode with ForwardDiff.jl\nin reverse mode with ChainRules.jl (WIP)","category":"page"},{"location":"#Inspirations","page":"Home","title":"Inspirations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"HMMBase.jl\nHMMGradients.jl\nControlledHiddenMarkovModels.jl","category":"page"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"warning: Work in progress\nIn the meantime, you can take a look at the files in test (especially test/correctness.jl) which demonstrate various ways in which the package can be used.","category":"page"}]
}
