using HiddenMarkovModels
using Documenter

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Maxime Mouchet, Guillaume Dalle and contributors",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md", "Notations" => "notations.md", "API reference" => "api.md"
    ],
    linkcheck=true,
    strict=false,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
