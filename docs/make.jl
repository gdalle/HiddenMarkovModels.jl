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

benchmarks_done = (
    length(readdir(joinpath(@__DIR__, "src", "assets", "benchmark", "plots"))) >
    1 &  # plots present
    length(readdir(joinpath(@__DIR__, "src", "assets", "benchmark", "results"))) >
    1  # results present
)

alt_pages = if benchmarks_done
    ["Features" => "alt_features.md", "Performance" => "alt_performance.md"]
else
    ["Features" => "alt_features.md"]
end

pages = [
    "Home" => "index.md",
    "Essentials" => ["Background" => "background.md", "API reference" => "api.md"],
    "Tutorials" => [
        "Built-in HMM" => "tuto_builtin.md",
        "Custom HMM" => "tuto_custom.md",
        "Debugging" => "debugging.md",
    ],
    "Alternatives" => alt_pages,
    "Advanced" => ["Formulas" => "formulas.md", "Roadmap" => "roadmap.md"],
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    repolink="https://github.com/gdalle/HiddenMarkovModels.jl",
    canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
    assets=String[],
)

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle",
    sitename="HiddenMarkovModels.jl",
    format=fmt,
    pages=pages,
    plugins=[bib],
    pagesonly=true,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
