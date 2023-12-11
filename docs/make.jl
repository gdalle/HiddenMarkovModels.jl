using Documenter
using DocumenterCitations
using HiddenMarkovModels
using Literate

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

examples_jl_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end

for file in readdir(examples_jl_path)
    Literate.markdown(joinpath(examples_jl_path, file), examples_md_path)
end

function literate_title(path)
    l = first(readlines(path))
    return l[3:end]
end

pages = [
    "Home" => "index.md",
    "API reference" => "api.md",
    "Tutorials" => [
        "Basics" => joinpath("examples", "basics.md"),
        "Interfaces" => joinpath("examples", "interfaces.md"),
        "Autodiff" => joinpath("examples", "autodiff.md"),
        "Time dependency" => joinpath("examples", "temporal.md"),
        "Control dependency" => joinpath("examples", "controlled.md"),
    ],
    "Advanced" => [
        "Alternatives" => "alternatives.md",
        "Debugging" => "debugging.md",
        "Formulas" => "formulas.md",
    ],
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
