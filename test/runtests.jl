using NCL
using ADNLPModels, NLPModels, NLPModelsTest

using SparseArrays
using Test

include("test_NCLModel.jl")
test_NCLModel()

include("test_NCLSolve.jl")
# test_NCLSolve()
