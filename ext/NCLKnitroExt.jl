module NCLKnitroExt

using NCL
using NLPModels
using KNITRO
using NLPModelsKnitro
using SolverCore

mutable struct KnitroNCLSubSolver <: AbstractNCLSubSolver
  solver::KnitroSolver
  stats::GenericExecutionStats
  dfeas_abs_tol::Float64  # KNITRO only supports Float64.
  pfeas_abs_tol::Float64
  mu_init::Float64  # just for logging
  name::String
  z::Vector{Float64}  # KNITRO wants a single vector of dual variables
end

# ... constructor
function NCL.KnitroNCLSubSolver(::NCLModel{T, S, M}; kwargs...) where {T, S, M}
  error("KNITRO only supports models with Float64 element type.")
end

function NCL.KnitroNCLSubSolver(
  ncl_model::NCLModel{Float64, S, M};
  dfeas_abs_tol::Float64 = 0.1,  # KNITRO stops when relative AND absolute tolerances are met.
  pfeas_abs_tol::Float64 = 0.1,  # Set them to loose values by default.
) where {S, M <: AbstractNLPModel{Float64, S}}
  @debug "initializing KNITRO subproblem solver"
  solver = KnitroSolver(ncl_model)
  stats = GenericExecutionStats(ncl_model)
  z = similar(get_x0(ncl_model))
  return KnitroNCLSubSolver(solver, stats, dfeas_abs_tol, pfeas_abs_tol, 0.0, "KNITRO", z)
end

const knitro_fixed_options = Dict(
  :nlp_algorithm => 1,       # Interior/Direct
  :bar_directinterval => 0,  # Only use direct linear algebra
  :bar_initpt => 3,          # Center initial guess wrt two-sided bounds
  :bar_murule => 1,          # Monotone
  :outlev => 0,
  :maxit => 100,
)

# TODO: need smarter initialization
function compute_mu_init(outer_iter::Int)
  mu_init = 1.0e-1
  if 2 <= outer_iter < 4
    mu_init = 1e-3
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
function (sub::KnitroNCLSubSolver)(
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
  bar_slackboundpush = sub.mu_init

  # warm-starting multipliers doesn't seem to help KNITRO
  y0 = sub.stats.multipliers
  zL0 = sub.stats.multipliers_L
  zU0 = sub.stats.multipliers_U
  sub.z .= zL0 .- zU0

  set_params!(
    sub.solver,
    x0 = x0,
    y0 = y0,
    z0 = sub.z,
    bar_initmu = sub.mu_init,
    bar_slackboundpush = bar_slackboundpush,
    opttol_abs = sub.dfeas_abs_tol,
    feastol_abs = sub.pfeas_abs_tol,
    opttol = rel_tol,
    feastol = rel_tol,
    knitro_fixed_options...,
    kwargs...,
  )

  return NLPModelsKnitro.solve!(sub.solver, ncl_model, sub.stats)
end

end
