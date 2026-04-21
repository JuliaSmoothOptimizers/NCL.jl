@testset "No solver by default" begin
  @test_throws ErrorException IpoptNCLSubSolver()
  @test_throws ErrorException KnitroNCLSubSolver()
  @test_throws ErrorException MadNLPNCLSubSolver()
end

@testset "Unconstrained problem errors out" begin
  f(x) = zero(eltype(x))
  x0 = [0.0]
  model = ADNLPModel(f, x0)
  @test_throws ErrorException NCLSolve(model)
end

@testset "Bound-constrained problem errors out" begin
  f(x) = zero(eltype(x))
  x0 = [0.0]
  lvar = [-1.0]
  uvar = [1.0]
  model = ADNLPModel(f, x0, lvar, uvar)
  @test_throws ErrorException NCLSolve(model)
end

using NLPModelsIpopt
using OptimizationProblems, OptimizationProblems.ADNLPProblems

@testset "IPOPT solver available" begin
  model = hs16()
  ncl_model = NCLModel(model)
  sub = IpoptNCLSubSolver(ncl_model)
  @test NCL.name(sub) == "IPOPT"
end

@testset "IPOPT solver only supports Float64" begin
  model = hs16(; type = Float32)
  ncl_model = NCLModel(model)
  @test_throws ErrorException IpoptNCLSubSolver(ncl_model)
end

@testset "Simple solve with IPOPT" begin
  model = hs16()
  ncl_model = NCLModel(model)
  subsolver = IpoptNCLSubSolver(ncl_model)
  stats = NCLSolve(ncl_model; subsolver = subsolver, verbose = false)
  @test stats.status == :first_order
end

@testset "Declare infeasibility at max penalty" begin
  # x is fixed at 0, while x + r == 100 forces a persistent residual r = 100.
  f(x) = zero(eltype(x))
  x0 = [0.0]
  lvar = [0.0]
  uvar = [0.0]
  c(x) = [x[1]]
  lcon = [100.0]
  ucon = [100.0]
  infeas_model = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon; name = "infeasible-rho-max")

  ncl_model = NCLModel(infeas_model; resid_linear = false)
  stats = NCLSolve(ncl_model, verbose = false)

  @test stats.status == :infeasible
  @test get(stats.solver_specific, :internal_msg, nothing) == :Solve_Failed
  @test maximum(abs, get(stats.solver_specific, :residuals, [0.0])) > 1.0e-6
end

using MadNLP

@testset "MadNLP solver available" begin
  model = hs16()
  ncl_model = NCLModel(model)
  sub = MadNLPNCLSubSolver(ncl_model)
  @test NCL.name(sub) == "MadNLP"
end

@testset "Simple solve with MadNLP" begin
  model = hs16()
  ncl_model = NCLModel(model)
  sub = MadNLPNCLSubSolver(ncl_model)
  stats = NCLSolve(ncl_model, subsolver = sub, verbose = false)
  @test stats.status == :first_order
end

using AmplNLReader

@testset "Simple TAX problem with IPOPT" begin
  model = AmplModel(joinpath(@__DIR__, "..", "data", "tax1D.nl"))
  ncl_model = NCLModel(model)
  stats = NCLSolve(ncl_model, verbose = false)
  @test stats.status == :first_order
end

@testset "Simple TAX problem with MadNLP" begin
  model = AmplModel(joinpath(@__DIR__, "..", "data", "tax1D.nl"))
  ncl_model = NCLModel(model)
  sub = MadNLPNCLSubSolver(ncl_model)
  stats = NCLSolve(ncl_model, subsolver = sub, verbose = false)
  @test stats.status == :first_order
end

@testset "Simple MPEC with IPOPT" begin
  model = AmplModel(joinpath(@__DIR__, "..", "data", "simplempec.nl"))
  ncl_model = NCLModel(model)
  stats = NCLSolve(ncl_model, verbose = false)
  @test stats.status == :first_order
end

@testset "Simple MPEC with MadNLP" begin
  model = AmplModel(joinpath(@__DIR__, "..", "data", "simplempec.nl"))
  ncl_model = NCLModel(model)
  sub = MadNLPNCLSubSolver(ncl_model)
  stats = NCLSolve(ncl_model, subsolver = sub, verbose = false)
  @test stats.status == :first_order
end
