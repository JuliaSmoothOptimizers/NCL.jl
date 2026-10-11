# NCL benchmarks

Scripts to measure how changes to NCL or its subproblem solvers affect a large set of problems.

## Problem sets

- `tax`: the optimal income tax problems shipped in `data/` (`tax1D`, `tax2D`, `pTax3D`, `pTax4D`, `pTax5D`), read with AmplNLReader.
- `cutest`: every constrained CUTEst problem within a size limit (`--max-var`, `--max-con`, both 100 by default), read with CUTEst.jl.
  CUTEst.jl downloads the SIF collection on first use and needs `gfortran`.

## Setup

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

The environment uses the NCL checkout in the parent directory, so results reflect whatever is checked out.
KNITRO needs a license; leave it out of `--solvers` if you do not have one.

## Running

```
julia --project=benchmark benchmark/run_benchmarks.jl --set tax --solvers ipopt,madnlp
julia --project=benchmark benchmark/run_benchmarks.jl --set cutest --solvers ipopt --max-var 200 --max-con 200
julia --project=benchmark benchmark/run_benchmarks.jl --set cutest --problems HS71,HS100
```

Each run writes `benchmark/results/<tag>_<set>.csv` with one row per (solver, problem): final status, status of the last subproblem, objective, residual norm ‖r‖∞, dual feasibility, outer and inner iterations, subproblem time and evaluation counts.
The tag defaults to the current git branch, and each row records the commit (with `-dirty` if `src/` or `ext/` has uncommitted changes).
`--max-time` (300 s by default) caps the total subproblem time, checked after each outer iteration, and each IPOPT or KNITRO subproblem, so a solve can overshoot it by at most one subproblem.
See the header of `run_benchmarks.jl` for all options.

## Comparing two versions

```
git checkout main
julia --project=benchmark benchmark/run_benchmarks.jl --set cutest
git checkout my-branch
julia --project=benchmark benchmark/run_benchmarks.jl --set cutest
julia --project=benchmark benchmark/compare.jl benchmark/results/main_cutest.csv benchmark/results/my-branch_cutest.csv
```

For each solver, `compare.jl` reports the number of problems solved, problems whose status changed, problems where both runs succeed but reach different objectives, and the shifted geometric mean of inner iterations, outer iterations and time over problems both runs solve.
