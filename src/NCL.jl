module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverCore
using KNITRO
using NLPModelsIpopt

global available_solvers = [:ipopt]

if KNITRO.has_knitro()
  using NLPModelsKnitro
  push!(available_solvers, :knitro)
end

"""
    _check_available_solver(solver::Symbol)

Return an error if `solver` is not in `NCL.available_solvers`
"""
function _check_available_solver(solver::Symbol)
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
