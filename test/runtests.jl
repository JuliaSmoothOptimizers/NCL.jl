using NCL
using ADNLPModels, NLPModels, NLPModelsTest

using SparseArrays
using Test

# TODO: update tests

@testset "Default solver loading" begin
  @test isempty(NCL.available_solvers)
end

include("test_NCLModel.jl")
test_NCLModel()

# include("test_NCLSolve.jl")
# test_NCLSolve()
