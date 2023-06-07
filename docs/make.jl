using Documenter
using HiddenMarkovModels
using Pkg

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

Pkg.activate(joinpath(@__DIR__, "..", "benchmark"))
include(joinpath(@__DIR__, "..", "benchmark", "plot_benchmarks.jl"))
Pkg.activate(@__DIR__)

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle, Maxime Mouchet and contributors",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Notations" => "notations.md",
        "Tutorial" => "tutorial.md",
        "API reference" => "api.md",
        "Benchmarks" => "benchmarks.md",
    ],
    linkcheck=true,
    strict=false,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
