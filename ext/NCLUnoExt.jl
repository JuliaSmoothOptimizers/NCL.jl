module NCLUnoExt

using NCL
using NLPModels
using SolverCore
using UnoSolver

mutable struct UnoNCLSubSolver <: AbstractNCLSubSolver
  stats::GenericExecutionStats
  dfeas_abs_tol::Float64
  pfeas_abs_tol::Float64
  compl_abs_tol::Float64
  mu_init::Float64
  name::String
end

function NCL.UnoNCLSubSolver(::NCLModel{T, S, M}; kwargs...) where {T, S, M}
  error("Uno only supports models with Float64 element type.")
end

function NCL.UnoNCLSubSolver(
  ncl_model::NCLModel{Float64, S, M};
  dfeas_abs_tol::Float64 = 0.1,
  pfeas_abs_tol::Float64 = 0.1,
  compl_abs_tol::Float64 = 0.1,
) where {S, M <: AbstractNLPModel{Float64, S}}
  @debug "initializing Uno subproblem solver"
  stats = GenericExecutionStats(ncl_model)
  return UnoNCLSubSolver(stats, dfeas_abs_tol, pfeas_abs_tol, compl_abs_tol, 0.0, "UNO")
end

# Uno does not use an interior-point barrier parameter directly, so keep this at 0 for logging.
compute_mu_init(::Int) = 0.0

function (sub::UnoNCLSubSolver)(
  ncl_model::NCLModel,
  outer_iter::Int,
  rel_tol::Float64;
  x0::AbstractVector = get_x0(ncl_model),
  kwargs...,
)
  sub.mu_init = compute_mu_init(outer_iter)

  # Warm start Uno by updating the initial primal/dual iterates on the NCL model.
  copyto!(get_x0(ncl_model), x0)
  copyto!(get_y0(ncl_model), sub.stats.multipliers)

  stationarity_tolerance = max(rel_tol, sub.dfeas_abs_tol)
  primal_feasibility_tolerance = max(rel_tol, sub.pfeas_abs_tol)
  tolerance = max(rel_tol, sub.compl_abs_tol)

  uno_stats = UnoSolver.uno(
    ncl_model;
    stationarity_tolerance = stationarity_tolerance,
    primal_feasibility_tolerance = primal_feasibility_tolerance,
    tolerance = tolerance,
    kwargs...,
  )

  sub.stats.status = uno_stats.status
  sub.stats.solution = uno_stats.solution
  sub.stats.objective = uno_stats.objective
  sub.stats.dual_feas = uno_stats.dual_feas
  sub.stats.primal_feas = uno_stats.primal_feas
  sub.stats.multipliers = uno_stats.multipliers
  sub.stats.iter = uno_stats.iter
  sub.stats.elapsed_time = uno_stats.elapsed_time

  if has_bounds(ncl_model)
    set_bounds_multipliers!(sub.stats, uno_stats.multipliers_L, uno_stats.multipliers_U)
  end

  return sub.stats
end

end
