using Documenter
using HiddenMarkovModels

DocMeta.setdocmeta!(
    HiddenMarkovModels, :DocTestSetup, :(using HiddenMarkovModels); recursive=true
)

benchmarks_successful = try
    include("process_benchmarks.jl")
    true
catch e
    @warn "Benchmarks were not processed" e
    false
end

pages = [
    "Home" => "index.md",
    "Background" => "background.md",
    "Tutorial" => "tutorial.md",
    "API reference" => "api.md",
    "Alternatives" => "alternatives.md",
    "Roadmap" => "roadmap.md",
]
if benchmarks_successful
    insert!(pages, length(pages) - 1, "Benchmarks" => "benchmarks.md")
end

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://gdalle.github.io/HiddenMarkovModels.jl",
    edit_link="main",
    assets=String[],
)

makedocs(;
    modules=[HiddenMarkovModels],
    authors="Guillaume Dalle, Maxime Mouchet and contributors",
    repo="https://github.com/gdalle/HiddenMarkovModels.jl/blob/{commit}{path}#{line}",
    sitename="HiddenMarkovModels.jl",
    format=fmt,
    pages=pages,
    linkcheck=true,
    strict=false,
)

deploydocs(; repo="github.com/gdalle/HiddenMarkovModels.jl", devbranch="main")
