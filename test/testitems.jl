using TestItemRunner

@testitem "Code quality" begin
    include("quality.jl")
end

@testitem "Code formatting" begin
    include("formatting.jl")
end

@testitem "Type stability" begin
    include("type_stability.jl")
end
#=
@testitem "Interface" begin
    include("interface.jl")
end
=#
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
