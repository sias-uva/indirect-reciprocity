using Pkg
using PackageCompiler

base_packages_to_compile = [
    "Revise",
    "OhMyREPL",
    "Debugger",
    "BenchmarkTools",
    "Cthulhu",
    "Documenter",
    "JuliaFormatter",
]

# Activate global environment and add (most up to date version of) packages
# above to global environment.
Pkg.activate()
Pkg.add(base_packages_to_compile)

create_sysimage(base_packages_to_compile; sysimage_path="BaseSysimage.dylib")

project_packages_to_compile = [
    "CairoMakie",
    "GLMakie",
    "Plots",
    "DataFrames",
    "ColorSchemes",
    "Format",
    "GeometryBasics",
    "StaticArrays",
    "StatsBase",
    "Tidier",
    "NonlinearSolve",
]

Pkg.activate("")
Pkg.add(project_packages_to_compile)

create_sysimage(
    project_packages_to_compile;
    sysimage_path="JuliaSysimage.dylib",
    base_sysimage="BaseSysimage.dylib",
)
