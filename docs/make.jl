using MPCC
using Documenter

DocMeta.setdocmeta!(MPCC, :DocTestSetup, :(using MPCC); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for file in readdir(joinpath(@__DIR__, "src")) if
  file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
  modules = [MPCC],
  authors = "Tangi Migot <tangi.migot@gmail.com>",
  repo = "https://github.com/tmigot/MPCC.jl/blob/{commit}{path}#{line}",
  sitename = "MPCC.jl",
  format = Documenter.HTML(; canonical = "https://tmigot.github.io/MPCC.jl"),
  pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/tmigot/MPCC.jl")
