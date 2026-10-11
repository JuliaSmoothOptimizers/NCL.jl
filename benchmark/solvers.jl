# NCL configurations to benchmark, one per subproblem solver.
#
# Each entry maps a solver name to `(nlp; max_time) -> GenericExecutionStats`.
# `max_time` bounds NCLSolve's total subproblem time, checked after each outer iteration,
# and, for IPOPT and KNITRO, each individual subproblem, so that a single hard problem
# cannot stall a sweep.

function ncl_ipopt(nlp; max_time = Inf, kwargs...)
  ncl = NCLModel(nlp)
  subsolver = IpoptNCLSubSolver(ncl)
  limits = isfinite(max_time) ? (max_wall_time = max_time,) : (;)
  return NCLSolve(ncl; subsolver, max_time, verbose = false, limits..., kwargs...)
end

# MadNLP 0.10 is not given max_wall_time: with max_wall_time = 300, it fails on tax2D
# with INVALID_NUMBER_HESSIAN_LAGRANGIAN before the first iteration. Each subproblem is
# still bounded by NCL's max_iter = 100 for MadNLP.
function ncl_madnlp(nlp; max_time = Inf, kwargs...)
  ncl = NCLModel(nlp)
  subsolver = MadNLPNCLSubSolver(ncl)
  return NCLSolve(ncl; subsolver, max_time, verbose = false, kwargs...)
end

function ncl_knitro(nlp; max_time = Inf, kwargs...)
  ncl = NCLModel(nlp)
  # AmplModel does not implement the linear/nonlinear constraint API.
  subsolver = KnitroNCLSubSolver(ncl; linear_api = !(nlp isa AmplModel))
  limits = isfinite(max_time) ? (maxtime = max_time,) : (;)
  return NCLSolve(ncl; subsolver, max_time, verbose = false, limits..., kwargs...)
end

const SOLVERS = Dict("ipopt" => ncl_ipopt, "madnlp" => ncl_madnlp, "knitro" => ncl_knitro)
