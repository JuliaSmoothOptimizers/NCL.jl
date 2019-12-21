using Logging

using AmplNLReader
# using NCL
# using NLPModelsIpopt
# using NLPModelsKnitro
using SolverTools
using SolverBenchmark

include("../src/NCL.jl")

# import KNITRO: KTR_OUTLEV_NONE, KN_OUTLEV_ITER_10
#
# ipopt_options = Dict(:tol => 1.0e-6,
#                      :dual_inf_tol => 1.0e-3,
#                      :constr_viol_tol => 1.0e-3,
#                      :max_iter => 500,
#                      :max_cpu_time => 1800.0,
#                      :mumps_mem_percent => 15,
#                      :print_level => 0,
#                      )
#
# knitro_options = Dict(:opttol => 1.0e-6,
#                       :opttol_abs => 1.0e-3,
#                       :feastol => 1.0e-6,
#                       :feastol_abs => 1.0e-3,
#                       :maxit => 500,
#                       :maxtime_cpu => 1800.0,
#                       :outlev => KTR_OUTLEV_NONE,
#                       )

probnames = ["tax1D", "tax2D", "pTax3D", "pTax4D", "pTax5D"]
problems = (AmplModel(probname) for probname in probnames)

solvers = Dict{Symbol,Function}(
               # :ipopt => prob -> ipopt(prob; ipopt_options...),
               # :knitro => prob -> knitro(prob; knitro_options...),
               :nclipopt => prob -> NCL.NCLSolve(prob, solver=:ipopt),
               :nclknitro => prob -> NCL.NCLSolve(prob, solver=:knitro),
               )

stats = bmark_solvers(solvers, problems)
