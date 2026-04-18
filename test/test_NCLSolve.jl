@testset "No solver by default" begin
  @test isempty(NCL.available_solvers)
end

using NLPModelsIpopt

@testset "IPOPT solver available" begin
  @test :ipopt ∈ NCL.available_solvers
end

using OptimizationProblems, OptimizationProblems.ADNLPProblems

@testset "Simple solve with IPOPT" begin
  model = hs16()
  ncl_model = NCLModel(model)
  stats = NCLSolve(ncl_model, solver = :ipopt, verbose = false)
  @test stats.status == :first_order
end
