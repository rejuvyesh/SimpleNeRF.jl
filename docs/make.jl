using SimpleNeRF
using Documenter

DocMeta.setdocmeta!(SimpleNeRF, :DocTestSetup, :(using SimpleNeRF); recursive=true)

makedocs(;
    modules=[SimpleNeRF],
    authors="rejuvyesh <mail@rejuvyesh.com> and contributors",
    repo="https://github.com/rejuvyesh/SimpleNeRF.jl/blob/{commit}{path}#{line}",
    sitename="SimpleNeRF.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/SimpleNeRF.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rejuvyesh/SimpleNeRF.jl",
    devbranch="main",
)
