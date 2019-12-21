# TODO: make NLPModelsKnitro optional

module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverTools
using NLPModelsIpopt
using NLPModelsKnitro

include("NCLModel.jl")
include("NCLSolve.jl")

end
