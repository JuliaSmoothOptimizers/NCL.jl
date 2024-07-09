# TODO: make NLPModelsKnitro optional

module NCL

using LinearAlgebra
using Printf

using NLPModels
using SolverCore
using KNITRO
using NLPModelsIpopt
using NLPModelsKnitro

include("NCLModel.jl")
include("NCLSolve.jl")

end
