module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverCore

const available_solvers = Symbol[]

function _register_solver!(solver::Symbol)
  solver in available_solvers || push!(available_solvers, solver)
  return available_solvers
end

function _solve_ipopt(ncl::Any; kwargs...)
  error(
    "Ipopt support is not loaded. Install and load NLPModelsIpopt.jl in your environment to use solver=:ipopt."
  )
end

function _solve_knitro(ncl::Any; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use solver=:knitro."
  )
end

"""
    _check_available_solver(solver::Symbol)

Return an error if `solver` is not in `NCL.available_solvers`
"""
function _check_available_solver(solver::Symbol)
  if isempty(available_solvers)
    error(
      "No NCL inner solver is available. Load NLPModelsIpopt.jl and/or KNITRO.jl + NLPModelsKnitro.jl."
    )
  end

  if !(solver in available_solvers)
    s = "`solver` must be one of these: "
    for x in available_solvers
      s *= "`$x`, "
    end
    error(s[1:(end - 2)])
  end
end

include("NCLModel.jl")
include("NCLSolve.jl")

end
