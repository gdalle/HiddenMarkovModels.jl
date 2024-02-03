using HMMComparison
using Random
using Test

rng = Random.default_rng()

@testset "HMMComparison" begin
    instance = Instance(;
        custom_dist=false, sparse=false, nb_states=5, obs_dim=10, seq_length=25, nb_seqs=10
    )
    params = build_params(rng, instance)
    data = build_data(rng, instance)
    logLs = compare_loglikelihoods(instance, params, data)
    for (key, val) in pairs(logLs)
        if key != "HMMBase.jl"
            if val isa AbstractVector
                @test all(val .≈ logLs["HiddenMarkovModels.jl"])
            elseif val isa Number
                @test val ≈ sum(logLs["HiddenMarkovModels.jl"])
            end
        end
    end
end
