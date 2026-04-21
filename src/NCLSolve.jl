export AbstractNCLSubSolver, NCLSolve

"""
    AbstractNCLSubSolver

Abstract type from which NCL subproblem solvers derive.
A subproblem solver must be a callable struct and should return an `AbstractExecutionStats` instance (that can be stored inside the solver and updated for efficiency).

NCL subproblem solvers conform to the following API:

## Constructor mandatory arguments

- `ncl_model::NCLModel`: the NCL model to be solved as a subproblem.

## Constructor keyword arguments
#
- `dfeas_abs_tol::eltype(ncl_model)`: the absolute tolerance on dual feasibility;
- `pfeas_abs_tol::eltype(ncl_model)`: the absolute tolerance on primal feasibility;
- `compl_abs_tol::eltype(ncl_model)`: the absolute tolerance on complementarity.

## Solver arguments

- `ncl_model::NCLModel`: the NCL model to be solved as a subproblem;
- `outer_iter::Int`: the outer iteration counter;
- `rel_tol::eltype(ncl_model)`: the relative stopping tolerance.

## Solver keyword arguments

- `x0::AbstractVector = get_x0(ncl_model)`: the initial guess.

All other keyword arguments will be passed directly to the underlying solver.
"""
abstract type AbstractNCLSubSolver <: AbstractOptimizationSolver end

name(sub::AbstractNCLSubSolver) = sub.name
stats(sub::AbstractNCLSubSolver) = sub.stats
mu_init(sub::AbstractNCLSubSolver) = sub.mu_init
elapsed_time(sub::AbstractNCLSubSolver) = sub.stats.elapsed_time

failed(stats::AbstractExecutionStats) = stats.status != :first_order

"""
    NCLSolve(ncl::AbstractNLPModel; kwargs...)
    NCLSolve(nlp::AbstractNLPModel; kwargs...)

Solve problem `ncl` using Algorithm NCL.
If `nlp` is not already an `NCLModel`, it will be converted to one.
If `nlp` does not have constraints other than bound constraints, an error will be thrown.
In that case, `nlp` should be passed directly to an unconstrained or bound-constrained solver.
It could also be passed directly to one of the NCL subproblem solvers that are currently supported.

The solver stops under the following circumstances:

1. The norm of the residual variables falls below `feas_tol` and the dual feasibility residual, as reported by the subproblem solver, falls below `opt_tol`;
2. the maximum number of outer iterations `max_iter_NCL` is attained;
3. the problem is found to be (locally) infeasible.

## Keyword arguments

- `opt_tol::Float64 = 1.0e-6`: dual feasibility stopping tolerance;
- `feas_tol::Float64 = 1.0e-6`: primal feasibility stopping tolerance;
- `max_iter_NCL::Int = 20`: maximum number of outer iterations;
- `subsolver::AbstractNCLSubSolver = IpoptNCLSubSolver(ncl)`: subproblem solver;
- `verbose::Bool = true`: whether to display progress information at each iteration.

Additional keyword arguments will be passed directly to the subproblem solver.
"""
function NCLSolve(nlp::AbstractNLPModel, args...; kwargs...)
  if unconstrained(nlp) || bound_constrained(nlp)
    error("NCL is designed for constrained problems only")
  end
  NCLSolve(NCLModel(nlp), args...; kwargs...)
end

function NCLSolve(
  ncl::NCLModel;
  opt_tol::Float64 = 1.0e-6,
  feas_tol::Float64 = 1.0e-6,
  max_iter_NCL::Int = 20,
  subsolver::AbstractNCLSubSolver = IpoptNCLSubSolver(ncl),
  verbose::Bool = true,
  kwargs...,  # will be passed directly to inner solver
)
  if verbose
    @info "NCL: using subsolver $(name(subsolver))"
  end

  NLPModels.reset!(ncl.nlp)
  NLPModels.reset!(ncl)

  nx = ncl.nx
  nr = ncl.nr

  τ_ρ = 10  # factor by which we increase ρ on unsuccessful iterations
  τ_η = 10  # factor by which we decrease η on successful iterations
  τ_ω = 10  # factor by which we decrease ω on successful iterations

  ncl.ρ = 1.0e+2
  ρ_max = 1.0e+12

  η = 1.0e+1  # initial primal feastibility tolerance
  ω0 = ω = 1.0e+1  # initial dual feasibility tolerance

  probname = replace(ncl.meta.name, "/" => "_")

  xr = copy(ncl.meta.x0)
  x = xr[1:nx]
  r = xr[(nx + 1):(nx + nr)]
  rNorm = norm(r, Inf)
  best_rNorm = rNorm
  y = ncl.meta.y0
  z = zeros(ncl.meta.nvar)

  # Initialize multipliers in subsolver.stats.
  # The subsolver uses these to warm start.
  sub_stats = stats(subsolver)
  SolverCore.reset!(sub_stats)
  set_multipliers!(sub_stats, y, z, z)

  if verbose
    @info @sprintf(
      "%5s  %5s  %9s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %6s",
      "outer",
      "inner",
      "NCL obj",
      "‖r‖",
      "η",
      "‖∇L‖",
      "ω",
      "ρ",
      "μ init",
      "‖y‖",
      "‖x‖",
      "time"
    )
  end

  k = 0
  t = 0.0
  iter_count = 0
  converged = false
  infeasible = false
  tired = k > max_iter_NCL

  while !(converged || infeasible || tired)
    k += 1

    # solve subproblem
    subsolver(ncl, k, ω; x0 = xr, kwargs...)

    failed(sub_stats) && @warn "subsolver returns with status " sub_stats.status

    xr = sub_stats.solution
    x = xr[1:nx]
    r = xr[(nx + 1):(nx + nr)]
    rNorm = norm(r, Inf)
    dual_feas = sub_stats.dual_feas
    inner = sub_stats.iter
    Δt = elapsed_time(subsolver)
    t += Δt

    iter_count += inner

    if verbose
      @info @sprintf(
        "%5d  %5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %6.2f",
        k,
        inner,
        obj(ncl, xr),
        rNorm,
        η,
        dual_feas,
        ω,
        ncl.ρ,
        mu_init(subsolver),
        norm(ncl.y, Inf),
        norm(x),
        Δt
      )
    end

    if rNorm ≤ max(η, feas_tol)
      ncl.y .+= ncl.ρ .* r
      η = η / τ_η
      ω = ω / τ_ω

    else
      ncl.ρ = min(ncl.ρ * τ_ρ, ρ_max)
      if ncl.ρ == ρ_max
        infeasible = !isfinite(rNorm) || rNorm > feas_tol
        if infeasible && isfinite(rNorm)
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " *
                string(ρ_max) *
                " reached at iteration k = " *
                string(k) *
                " with residual norm " *
                string(rNorm) *
                ", declaring infeasibility."
        elseif infeasible
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " *
                string(ρ_max) *
                " reached at iteration k = " *
                string(k) *
                " with non-finite residual norm, declaring infeasibility."
        else
          @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " *
                string(ρ_max) *
                " reached at iteration k = " *
                string(k) *
                " with small residual norm " *
                string(rNorm) *
                ", not declaring infeasibility."
        end
      end
    end

    converged = rNorm ≤ feas_tol && dual_feas ≤ opt_tol
    tired = k > max_iter_NCL
  end

  if converged
    status = :first_order
  elseif infeasible
    status = :infeasible
  elseif tired
    status = :max_iter
  else
    status = sub_stats.status
  end
  dual_feas = sub_stats.dual_feas
  primal_feas = η
  if has_bounds(ncl.nlp)
    zL = sub_stats.multipliers_L[1:nx]
    zU = sub_stats.multipliers_U[1:nx]
  end

  ncl_stats = GenericExecutionStats(
    ncl.nlp,
    status = status,
    solution = x,
    iter = iter_count,
    primal_feas = primal_feas,
    dual_feas = dual_feas,
    objective = obj(ncl.nlp, x),
    elapsed_time = t,
    multipliers = ncl.y,
    #! doesn't work... counters = nlp.counters,
    solver_specific = Dict(
      :internal_msg => converged ? :Solve_Succeeded : :Solve_Failed,
      :residuals => r,
    ),
  )
  if has_bounds(ncl.nlp)
    set_bounds_multipliers!(ncl_stats, zL, zU)
  end
  return ncl_stats
end
