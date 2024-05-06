# Indirect Reciprocity (IR)

This repository contains the source code for my work on the (algorithmic) game theory topic of indirect reciprocity: a mechanism for encouraging cooperation between self-interested agents.
The work focuses on the interplay between cooperation and fairness, investigating how we can engineer systems of indirect reciprocity, whether artificial or otherwise, to achieve fair, socially desirable outcomes.

The repository will receive updates following the formal publication of any future work on the topic.

If this is your first time looking at a Julia repository, read the [New to Julia?](#new-to-julia) section for guidance on where and how to find what you're looking for.

### What is indirect reciprocity?

Indirect reciprocity allows agents to observe (directly or indirectly) and judge the actions of others, then use these judgements in determining whether to cooperate or defect with an individual in the future.
This allows agents who have never interacted before to make better-informed decisions taking into account the other agent's reputation.

What we observe in real life is that reputation mechanisms are presented with other information that agents use when making decisions.
For example, someone's name on a job application may be used to qualify an applicant's credentials, or, on a site such as Airbnb, a host may discriminate based on race.

This work combines the mechanisms of homophily (or more broadly, tag-based cooperation) and indirect reciprocity to see how fair cooperation can be achieved, and the barriers to its realisation that may be present in social systems.

### Papers and presentations related to this project
- The visionary/future outlook paper [Learning Fair Cooperation in Systems of Indirect Reciprocity](https://alaworkshop2023.github.io/papers/ALA2023_paper_53.pdf) has been accepted to the [Adaptive and Learning Agents workshop](https://alaworkshop2023.github.io) at [AAMAS2023](https://aamas2023.soton.ac.uk/).

## Structure of the repository
This is a "monorepo" with the `packages` folder containing three Julia modules `IR` and `RL`, `IRUtils`.
- `IR` contains the core source code to determine the reputations and payoffs in an evolutionary game theory (EGT) model of tag-based and reputation-based cooperation. As the science machine is cranked, this will be made into a stand-alone repository.
- `RL` uses `IR`'s definition of `Norm`, `Agent`, and its linear interpolation functions `lerp` and `mistake` to define a basic agent-based model representation of `IR`'s EGT model with Q-learning in place of "social learning".
- `IRUtils` contains a number of functions used in multiple scripts
- `projects` contains folders for each paper, the scripts in the folders allow one to reproduce the data and figures found in the article and supplementary information. To aid this, each project also contains the Project.toml and Manifest.toml files from the last time the paper was worked on.

## How to run the code
Inside each `projects` subfolder, there will be specific instructions on how to reproduce every figure from the corresponding paper.
These instructions assume that Julia is installed and all the required packages are available.
To do this:
1. Install Julia with [juliaup](https://github.com/JuliaLang/juliaup) or otherwise.
    a. Until `Pkg` is updated, delete IR, RL, and IRUtils (and their `[compat]` entries) from the Project.toml and run `]dev --local packages/IR packages/RL packages/IRUtils`.
2. Install the required packages by opening Julia with this repository as its home directory, and running 
```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

The Manifest.toml ensures that the package versions you install are identical to the ones used to generate this.
If you encounter issues when running the code, try replacing the home directory versions with the project-specific Manifest.toml and Project.toml files found in the corresponding `toml` directory.

## New to Julia?

Inside this repository are two Julia packages, `IR` and `RL`.
By my own definition, a Julia package is a Julia module, which is a collection of Julia source code, and some extra files to help it stand on its own two feet.
The module itself is found in (for example) `ModuleName.jl`, declared by `module ModuleName` and ended at the bottom of the file with `end`, and serves as the jumping off point for anyone looking to see what a module is for, which functions are intended to for end-users, and (more recently through `PrecompileTools.jl`) what an example workload of the module looks like.

A package containing a nontrivial module will also typically have dependencies on other packages.
These, along with the package's name and version can be found in the `Project.toml` in the same folder that `src/ModuleName.jl` lives.
The `Project.toml`, along with the machine-generated `Manifest.toml` ensure that the package can be installed on any system that has a Julia version meeting the compatability requirements specified by the `Project.toml`.

Finally, some packages also include a `.JuliaFormatter.toml` file which specifies how the package should be formatter by `JuliaFormatter.jl`. The codestyle used by this repository is [BlueStyle](https://github.com/invenia/BlueStyle), but a commonly used one is [SciMLStyle](https://github.com/SciML/SciMLStyle) which is far more restrictive, doesn't look as nice, but is meant to minimise possible [footguns](https://en.wiktionary.org/wiki/footgun).

### General tips when reading Julia code
- Filenames written in CamelCase contain either module or type definitions, those in snake_case may contain arbitrary code.
- In `MyModule.jl`, you'll find `export` and `@reexport` statements which define what the module chooses to make public. These functions/types are intended to be used by end users.
- New types (classes, if you prefer to think of them as such) are declared with `struct` or `mutable struct` and are given capital letter names. In idiomatic code, if you see a capital letter, you can assume it's type related.
- New functions can be declared with the keyword `function functionname(arguments); stuff...; end` or simply `functionname(arguments) = stuff...` in one line.
- Function arguments following a `;` are keyword only, those preceding are positional only.
