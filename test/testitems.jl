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

@testitem "Correctness" begin
    include("correctness.jl")
end

@testitem "SparseArrays" begin
    include("sparse.jl")
end

@testitem "LogarithmicNumbers" begin
    include("logarithmic.jl")
end

@testitem "ForwardDiff" begin
    include("forwarddiff.jl")
end

@testitem "Zygote" begin
    include("zygote.jl")
end
