## Milk POWDER (II): the milk Production Optimizer incorporating Weather Dynamics and Econmic risk (II)

This repository implements the model described in Dowson et al. (2017). *A
multi-stage stochastic optimization model of a dairy farm with spot-price
uncertainty*. Manuscript in preparation.

The easiest way to run it is via the command line

    julia POWDER.jl "model.parameters.json"

##### Installation Instructions

Install [Julia v0.6.0](https://github.com/JuliaLang/julia/releases/tag/v0.6.0) and [Gurobi v0.7.0](http://gurobi.com/).

Once Julia is installed,the following packages are necessary to replicate the
results found in the paper:
    - SDDP.jl at commit [b7c3e3f](https://github.com/odow/SDDP.jl/commit/b7c3e3fc5c17d53c47ca2bde45fd5f526126d2ef)
    - JuMP.jl [v0.18.0](https://github.com/JuliaOpt/JuMP.jl/releases/tag/v0.18.0)
    - Gurobi.jl [v0.3.3](https://github.com/JuliaOpt/JuMP.jl/releases/tag/v0.3.3)
    - JSON.jl [v0.13.0](https://github.com/JuliaIO/JSON.jl/releases/tag/v0.13.0)
