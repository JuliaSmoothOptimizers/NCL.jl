# NCL

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/NCL.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/NCL.jl/dev)
[![Test workflow status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSmoothOptimizers/NCL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/NCL.jl)
[![Lint workflow Status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/JuliaSmoothOptimizers/NCL.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

An implementation of [Algorithm NCL](https://dx.doi.org/10.1007/978-3-319-90026-1_8) in pure Julia using either [IPOPT](https://coin-or.github.io/Ipopt) or [Artelys KNITRO](https://www.artelys.com/knitro) to solve the subproblems.

The `data` folder contains several tax models in [AMPL](http://www.ampl.com) format that can be read with [AmplNLReader](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl).
Any model complying with the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API can be passed to the NCL solver, e.g., those from the [CUTEst](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) collection.

The solver is still work in progress but is functional.
A similar solver is available for tax problems only in the AMPL scripting language: <https://github.com/optimizers/ncl>.

## References

- D. Ma, Judd, K., Orban, D., & Saunders, M. (2018). [Stabilized optimization via an NCL algorithm](https://dx.doi.org/10.1007/978-3-319-90026-1_8). In M. Al-Baali, L. Grandinetti, & A. Purnama (Eds.), Numerical Analysis and Optimization (Vol. 235, pp. 173–191). Switzerland: Springer International Publishing.
- Several talks by Michael A. Saunders: <https://stanford.edu/group/SOL/publications_talks.html>
