export NCLSolve

const ipopt_fixed_options = Dict(:sb => "yes",  # options that are always used
                                 :print_level => 0,
                                 :max_iter => 100,
                                 )

const knitro_fixed_options = Dict(:algorithm => 1,
                                  :bar_directinterval => 0,
                                  :bar_initpt => 2,
                                  :outlev => 0,
                                  :maxit => 100,
                                  )

NCLSolve(nlp::AbstractNLPModel, args...; kwargs...) = NCLSolve(NCLModel(nlp), args...; kwargs...)


function NCLSolve(ncl::NCLModel;
                  opt_tol::Float64=1.0e-6,
                  feas_tol::Float64=1.0e-6,
                  max_iter_NCL::Int = 20,
                  solver = KNITRO.has_knitro() ? :knitro : :ipopt,
                  kwargs...  # will be passed directly to inner solver
                 )

    nx = ncl.nx
    nr = ncl.nr

    mu_init = 1.0e-1

    τ_ρ = 10  # factor by which we increase ρ on unsuccessful iterations
    τ_η = 10  # factor by which we decrease η on successful iterations
    τ_ω = 10  # factor by which we decrease ω on successful iterations

    ncl.ρ = 1.0e+2
    ρ_max = 1.0e+12

    η = 1.0e+1  # initial primal feastibility tolerance
    ω0 = ω = 1.0e+1  # initial dual feasibility tolerance

    probname = replace(ncl.meta.name, "/" => "_")

    xr = copy(ncl.meta.x0)
    x = xr[1 : nx]
    r = xr[nx + 1 : nx + nr]
    rNorm = norm(r, Inf)
    best_rNorm = rNorm
    y = ncl.meta.y0
    z = zeros(ncl.meta.nvar)

    @info @sprintf("%5s  %5s  %9s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %6s",
                   "outer", "inner", "NCL obj", "‖r‖", "η", "‖∇L‖", "ω", "ρ", "μ init", "‖y‖", "‖x‖", "time")

    k = 0
    t = 0.0
    iter_count = 0
    converged = false
    tired = k > max_iter_NCL

    # set absolute tolerances once and for all
    ipopt_options = Dict{Symbol,Any}(:dual_inf_tol => ω,     # leave these at the initial ω value
                                     :constr_viol_tol => ω,
                                    );

    knitro_options = Dict{Symbol,Any}(#:opttol_abs => ω,  # for some reason this doesn't work?!?!?
                                      #:feastol_abs => ω,
                                     );

    local inner_stats, z, zL, zU

    while !(converged || tired)
        k += 1

        if solver == :ipopt
          if k == 2
              mu_init = 1e-4
          elseif k == 4
              mu_init = 1e-5
          elseif k == 6
              mu_init = 1e-6
          elseif k == 8
              mu_init = 1e-7
          elseif k == 10
              mu_init = 1e-8
          end

          inner_stats = ipopt(ncl;
                              x0 = xr,
                              warm_start_init_point = k > 1 ? "yes" : "no",
                              mu_init = mu_init,
                              tol = ω,
                              ipopt_fixed_options...,
                              ipopt_options...,
                              kwargs...)

          # warm-starting multipliers appears to help IPOPT
          ipopt_options[:y0]  = inner_stats.solver_specific[:multipliers_con]
          ipopt_options[:zL0] = inner_stats.solver_specific[:multipliers_L]
          ipopt_options[:zU0] = inner_stats.solver_specific[:multipliers_U]

        elseif solver == :knitro

          if k == 2
            mu_init = 1e-3
            knitro_options[:bar_slackboundpush] = 1.0e-3
            knitro_options[:bar_murule] = 1
          elseif k == 4
            mu_init = 1e-5
            knitro_options[:bar_slackboundpush] = 1.0e-5
          elseif k == 6
            mu_init = 1e-6
            knitro_options[:bar_slackboundpush] = 1.0e-6
          elseif k == 8
            mu_init = 1e-7
            knitro_options[:bar_slackboundpush] = 1.0e-7
          elseif k == 10
            mu_init = 1e-8
            knitro_options[:bar_slackboundpush] = 1.0e-8
          end
          knitro_options[:bar_initmu] = mu_init
          knitro_options[:opttol] = ω
          knitro_options[:feastol] = ω
          knitro_options[:opttol_abs] = ω0
          knitro_options[:feastol_abs] = ω0
          inner_stats = knitro(ncl;
                               x0 = xr,
                               knitro_fixed_options...,
                               knitro_options...,
                               kwargs...)

          # warm-starting multipliers doesn't seem to help KNITRO
          # knitro_options[:y0] = inner_stats.solver_specific[:multipliers_con]
          # knitro_options[:z0] = inner_stats.solver_specific[:multipliers_L]
        else
          error("The solver $solver is not supported.")
        end

        inner_stats.status == :first_order || @warn "inner solver returns with status" inner_stats.status

        xr = inner_stats.solution
        x = xr[1:nx]
        r = xr[nx+1 : nx+nr]
        rNorm = norm(r, Inf)
        dual_feas = inner_stats.dual_feas
        inner = inner_stats.iter
        Δt = inner_stats.elapsed_time
        t += Δt

        iter_count += inner

        @info @sprintf("%5d  %5d  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %6.2f",
                       k, inner, obj(ncl, xr), rNorm, η, dual_feas, ω, ncl.ρ, mu_init, norm(ncl.y, Inf), norm(x), Δt)

        if rNorm ≤ max(η, feas_tol)
          ncl.y .+= ncl.ρ * r
          η = η / τ_η
          ω = ω / τ_ω

        else
          ncl.ρ = min(ncl.ρ*τ_ρ, ρ_max)
          if ncl.ρ == ρ_max
            @warn "\nin NCLSolve($(ncl.nlp.meta.name)): maximum penalty ρ = " * string(ρ_max) * " reached at iteration k = " * string(k)
          end
        end

        converged = rNorm ≤ feas_tol && dual_feas ≤ opt_tol
        tired = k > max_iter_NCL
    end

    if converged
      status = :first_order
    elseif tired
      status = :max_iter
    else
      status = inner_stats.status
    end
    dual_feas = inner_stats.dual_feas
    primal_feas = η
    zL = inner_stats.solver_specific[:multipliers_L][1:nx]
    # zU = inner_stats.solver_specific[:multipliers_U][1:nx]  # doesn't work with KNITRO
    zU = inner_stats.solver_specific[:multipliers_U]

    return GenericExecutionStats(status, ncl,
                                 solution = x,
                                 iter = iter_count,
                                 primal_feas = primal_feas,
                                 dual_feas = dual_feas,
                                 objective = obj(ncl.nlp, x),
                                 elapsed_time = t,
                                 #! doesn't work... counters = nlp.counters,
                                 solver_specific = Dict(:multipliers_con => ncl.y,
                                                        :multipliers_L => zL,
                                                        :multipliers_U => zU,
                                                        :internal_msg => converged ? :Solve_Succeeded : :Solve_Failed,
                                                        :residuals => r
                                                      )
                                )

end
