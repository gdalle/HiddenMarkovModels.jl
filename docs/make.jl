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

benchmarks_done = false

pages = [
    "Home" => "index.md",
    "Essentials" => ["Background" => "background.md", "API reference" => "api.md"],
    "Tutorials" => [
        "Built-in HMM" => "builtin.md",
        "Custom HMM" => "custom.md",
        "Debugging" => "debugging.md",
    ],
    "Alternatives" => if benchmarks_done
        ["Features" => "features.md", "Benchmarks" => "benchmarks.md"]
    else
        ["Features" => "features.md"]
    end,
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
