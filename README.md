# NCL

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/NCL.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/NCL.jl/dev)
[![Test workflow status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSmoothOptimizers/NCL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/NCL.jl)
[![Lint workflow Status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

An implementation of [Algorithm NCL](https://dx.doi.org/10.1007/978-3-319-90026-1_8) in pure Julia.
NCL currently supports the following subproblem solvers:

- [IPOPT](https://coin-or.github.io/Ipopt)
- [Artelys KNITRO](https://www.artelys.com/knitro)
- [MadNLP](https://github.com/MadNLP/MadNLP.jl)

The `data` folder contains several of the original tax models in [AMPL](http://www.ampl.com) format that can be read with [AmplNLReader](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl).
Any model complying with the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API can be passed to the NCL solver, e.g., those from the [CUTEst](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) collection, or the pure Julia models of [OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl).

The solver is still work in progress but is functional.

## Historical Notes

Our first NCL solver was written directly in the AMPL scripting language and was restricted to solving the tax problems.
It is available from <https://github.com/optimizers/ncl>.

## References

- Ma, D., Judd, K., Orban, D., & Saunders, M. (2018). [Stabilized optimization via an NCL algorithm](https://dx.doi.org/10.1007/978-3-319-90026-1_8). In M. Al-Baali, L. Grandinetti, & A. Purnama (Eds.), Numerical Analysis and Optimization (Vol. 235, pp. 173–191). Switzerland: Springer International Publishing.
- Ma, D., Orban, D., & Saunders, M.A. (2021). [A Julia Implementation of Algorithm NCL for Constrained Optimization](https://doi.org/10.1007/978-3-030-72040-7_8). In: Al-Baali, M., Purnama, A., Grandinetti, L. (eds) Numerical Analysis and Optimization. NAO 2020. Springer Proceedings in Mathematics & Statistics, vol 354. Springer, Cham.
- Ma, D., Orban, D. & Saunders, M.A. (2025). [Solving Algorithm NCL’s Subproblems: The Need for Interior Methods](https://doi.org/10.1007/s10013-025-00760-z). Vietnam J. Math. 53, 915–919.
- Several talks by Michael A. Saunders: <https://stanford.edu/group/SOL/publications_talks.html>
