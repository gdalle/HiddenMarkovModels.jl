using Documenter
using DocumenterCitations
using HiddenMarkovModels

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

pages = [
    "Home" => "index.md",
    "Background" => "background.md",
    "Tutorial" => "tutorial.md",
    "Alternatives" => "alternatives.md",
    "Benchmarks" => "benchmarks.md",
    "Roadmap" => "roadmap.md",
    "API reference" => "api.md",
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
    edit_link="main",
    assets=String[],
)

makedocs(
    bib;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle, Maxime Mouchet and contributors",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=fmt,
    pages=pages,
    linkcheck=false,
    strict=false,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
