module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverCore

export IpoptNCLSubSolver
export KnitroNCLSubSolver
export MadNLPNCLSubSolver

include("NCLModel.jl")
include("NCLSolve.jl")

@doc (@doc AbstractNCLSubSolver) IpoptNCLSubSolver
function IpoptNCLSubSolver(args...; kwargs...)
  error(
    "Ipopt support is not loaded. Install and load NLPModelsIpopt.jl in your environment to use IpoptNCLSubSolver.",
  )
end

@doc (@doc AbstractNCLSubSolver) KnitroNCLSubSolver
function KnitroNCLSubSolver(args...; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use KnitroNCLSubSolver.",
  )
end

@doc (@doc AbstractNCLSubSolver) MadNLPNCLSubSolver
function MadNLPNCLSubSolver(args...; kwargs...)
  error(
    "MadNLP support is not loaded. Install and load MadNLP.jl in your environment to use MadNLPNCLSubSolver.",
  )
end

end
