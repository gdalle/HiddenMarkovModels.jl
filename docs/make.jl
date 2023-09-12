using Documenter
using DocumenterCitations
using HiddenMarkovModels

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

open(joinpath(joinpath(@__DIR__, "src"), "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/gdalle/HiddenMarkovModels.jl/blob/main/README.md"
        ```
        """,
    )
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

pages = [
    "Home" => "index.md",
    "Essentials" => ["Background" => "background.md", "API reference" => "api.md"],
    "Tutorials" => [
        "Built-in HMM" => "tuto_builtin.md",
        "Custom HMM" => "tuto_custom.md",
        "Debugging" => "debugging.md",
    ],
    "Alternatives" =>
        ["Features" => "alt_features.md", "Performance" => "alt_performance.md"],
    "Advanced" => ["Formulas" => "formulas.md", "Roadmap" => "roadmap.md"],
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
    authors="Guillaume Dalle",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=fmt,
    pages=pages,
    linkcheck=false,
    strict=false,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
