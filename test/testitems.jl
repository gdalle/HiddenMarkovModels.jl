# Don't run this on Julia < 1.9, it will explode

using TestItemRunner

@testitem "Code formatting" begin
    include("formatting.jl")
end

@testitem "Code quality" begin
    include("quality.jl")
end

@testitem "Code linting" begin
    include("linting.jl")
end

@testitem "Interface" begin
    include("interface.jl")
end

@testitem "Type stability" begin
    include("type_stability.jl")
end

@testitem "Correctness" begin
    include("correctness.jl")
end

@testitem "Sparse" begin
    include("sparse.jl")
end

@testitem "Static" begin
    include("static.jl")
end

@testitem "ForwardDiff" begin
    include("forwarddiff.jl")
end

@testset "Doctests" begin
    include("doctests.jl")
end
