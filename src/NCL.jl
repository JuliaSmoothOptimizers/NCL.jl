module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverCore

export IpoptNCLSubSolver
export KnitroNCLSubSolver

function IpoptNCLSubSolver(args...; kwargs...)
  error(
    "Ipopt support is not loaded. Install and load NLPModelsIpopt.jl in your environment to use IpoptNCLSubSolver.",
  )
end

function KnitroNCLSubSolver(args...; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use KnitroNCLSubSolver.",
  )
end

include("NCLModel.jl")
include("NCLSolve.jl")

end
