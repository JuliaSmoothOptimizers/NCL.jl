module NCLIpoptExt

using NCL
using NLPModels
using NLPModelsIpopt
using SolverCore

mutable struct IpoptNCLSubSolver <: AbstractNCLSubSolver
  solver::IpoptSolver
  stats::GenericExecutionStats
  dfeas_abs_tol::Float64  # IPOPT only supports Float64.
  pfeas_abs_tol::Float64
  compl_abs_tol::Float64
  mu_init::Float64  # just for logging
  name::String
end

# ... constructor
function NCL.IpoptNCLSubSolver(::NCLModel{T, S, M}; kwargs...) where {T, S, M}
  error("IPOPT only supports models with Float64 element type.")
end

function NCL.IpoptNCLSubSolver(
  ncl_model::NCLModel{Float64, S, M};
  dfeas_abs_tol::Float64 = 0.1,  # IPOPT stops when relative AND absolute tolerances are met.
  pfeas_abs_tol::Float64 = 0.1,  # Set them to loose values by default.
  compl_abs_tol::Float64 = 0.1,
) where {S, M <: AbstractNLPModel{Float64, S}}
  @debug "initializing IPOPT subproblem solver"
  solver = IpoptSolver(ncl_model)
  stats = GenericExecutionStats(ncl_model)
  return IpoptNCLSubSolver(solver, stats, dfeas_abs_tol, pfeas_abs_tol, compl_abs_tol, 0.0, "IPOPT")
end

const ipopt_fixed_options = Dict(
  :sb => "yes",  # options that are always used
  :print_level => 0,
  :max_iter => 100,
)

# TODO: need smarter initialization
function compute_mu_init(outer_iter::Int)
  mu_init = 1.0e-1
  if 2 <= outer_iter < 4
    mu_init = 1e-4
  elseif 4 <= outer_iter < 6
    mu_init = 1e-5
  elseif 6 <= outer_iter < 8
    mu_init = 1e-6
  elseif 8 <= outer_iter < 10
    mu_init = 1e-7
  elseif outer_iter >= 10
    mu_init = 1e-8
  end
  mu_init
end

# ... solve
function (sub::IpoptNCLSubSolver)(
  ncl_model::NCLModel,
  outer_iter::Int,
  rel_tol::Float64;
  x0::AbstractVector = get_x0(ncl_model),
  kwargs...,
)

  # prepare for warm start
  # TODO: try solver.mu from the previous solve
  # TODO: set bound_push?
  sub.mu_init = compute_mu_init(outer_iter)

  # warm-starting multipliers appears to help IPOPT
  y0 = sub.stats.multipliers
  zL0 = sub.stats.multipliers_L
  zU0 = sub.stats.multipliers_U
  return NLPModelsIpopt.solve!(
    sub.solver,
    ncl_model,
    sub.stats;
    warm_start_init_point = outer_iter > 1 ? "yes" : "no",
    x0 = x0,
    y0 = y0,
    zL0 = zL0,
    zU0 = zU0,
    mu_init = sub.mu_init,
    dual_inf_tol = sub.dfeas_abs_tol,
    constr_viol_tol = sub.pfeas_abs_tol,
    compl_inf_tol = sub.compl_abs_tol,
    tol = rel_tol,
    ipopt_fixed_options...,
    kwargs...,
  )
end

end
