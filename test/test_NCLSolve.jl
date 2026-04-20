@testset "No solver by default" begin
  @test_throws ErrorException IpoptNCLSubSolver()
  @test_throws ErrorException KnitroNCLSubSolver()
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
  stats = NCLSolve(ncl_model, solver = :ipopt, verbose = false)

  @test stats.status == :infeasible
  @test get(stats.solver_specific, :internal_msg, nothing) == :Solve_Failed
  @test maximum(abs, get(stats.solver_specific, :residuals, [0.0])) > 1.0e-6
end
