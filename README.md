# NCL.jl

An implementation of [Algorithm NCL](https://dx.doi.org/10.1007/978-3-319-90026-1_8) in pure Julia using either [IPOPT](http://www.coin-or.org/Ipopt/documentation/documentation.html) or [Artelys KNITRO](https://www.artelys.com/knitro) to solve the subproblems.

The `data` folder contains several tax models in [AMPL](http://www.ampl.com) format that can be read with [AmplNLReader](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl).
Any model complying with the [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API can be passed to the NCL solver, e.g., those from the [CUTEst](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) collection.

The solver is still work in progress but is functional.
A similar solver is available for tax problems only in the AMPL scripting language: https://github.com/optimizers/ncl.

### References

* D. Ma, Judd, K., Orban, D., & Saunders, M. (2018). [Stabilized optimization via an NCL algorithm](https://dx.doi.org/10.1007/978-3-319-90026-1_8). In M. Al-Baali, L. Grandinetti, & A. Purnama (Eds.), Numerical Analysis and Optimization (Vol. 235, pp. 173–191). Switzerland: Springer International Publishing.
* Several talks by Michael A. Saunders: http://stanford.edu/group/SOL/talks.html
