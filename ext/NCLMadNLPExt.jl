module NCLMadNLPExt

using MadNLP

using NCL
using NLPModels
using SolverCore

SolverCore.reset!(::MadNLP.MadNLPExecutionStats) = nothing
# MadNLPExecutionStats starts with uninitialized multipliers; NCLSolve sets them before the first solve.
function SolverCore.set_multipliers!(stats::MadNLP.MadNLPExecutionStats, y, zL, zU)
  copyto!(stats.multipliers, y)
  copyto!(stats.multipliers_L, zL)
  copyto!(stats.multipliers_U, zU)
  return stats
end

mutable struct MadNLPNCLSubSolver{T <: Real} <: AbstractNCLSubSolver
  solver::MadNLPSolver
  stats::MadNLP.MadNLPExecutionStats
  mu_init::T  # just for logging
  name::String
end

# ... constructor
function NCL.MadNLPNCLSubSolver(ncl_model::NCLModel; kwargs...)
  @debug "initializing MadNLP subproblem solver"
  solver = MadNLPSolver(ncl_model, print_level = MadNLP.ERROR; kwargs...)
  stats = MadNLP.MadNLPExecutionStats(solver)
  return MadNLPNCLSubSolver(solver, stats, zero(eltype(ncl_model)), "MadNLP")
end

# MadNLPExecutionStats has no elapsed_time field; read from the solver's counter instead.
NCL.elapsed_time(sub::MadNLPNCLSubSolver) = sub.solver.cnt.total_time

# MadNLPExecutionStats.status is a MadNLP.Status enum, not a Julia symbol.
NCL.failed(stats::MadNLP.MadNLPExecutionStats) =
  stats.status ∉ (MadNLP.SOLVE_SUCCEEDED, MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL)

const madnlp_fixed_options = Dict(:max_iter => 100, :dual_initialized => true)

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

# MadNLP 0.10 moved mu_init into opt.barrier and silently ignores it as a solve! keyword.
set_mu_init!(opt, mu_init) =
  hasproperty(opt, :barrier) ? (opt.barrier.mu_init = mu_init) : (opt.mu_init = mu_init)

# ... solve
function (sub::MadNLPNCLSubSolver)(
  ::NCLModel,  # MadNLP stores the problem inside the solver; this argument is only here for compatibility with the API
  outer_iter::Int,
  rel_tol::Float64;
  x0::AbstractVector = get_x0(sub.solver.nlp),
  kwargs...,
)

  # prepare for warm start
  # Restarting from the previous solve's final mu does not help: with the loose subproblem
  # tolerances, the solver stops before decreasing mu, so it ends at the mu_init it started from.
  sub.mu_init = compute_mu_init(outer_iter)
  set_mu_init!(sub.solver.opt, sub.mu_init)
  bound_push = sub.mu_init  # only used by MadNLP on the first solve

  # MadnLP uses info from the problem itself to warm start.
  # The problem is stored inside the solver.
  copyto!(get_x0(sub.solver.nlp), x0)  # to warm start the next outer iteration
  copyto!(get_y0(sub.solver.nlp), sub.stats.multipliers)

  # MadNLP never resets its counters on a re-solve: without this, max_iter, max_wall_time,
  # the reported iteration count and elapsed time all accumulate across outer iterations.
  sub.solver.cnt.k = 0
  sub.solver.cnt.start_time = time()

  return MadNLP.solve!(
    sub.solver,
    sub.stats;
    bound_push = bound_push,
    tol = rel_tol,
    madnlp_fixed_options...,
    kwargs...,
  )
end

end
