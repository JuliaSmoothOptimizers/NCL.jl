# Run NCL on a problem set with one or more subproblem solvers and save the results to CSV.
#
#   julia --project=benchmark benchmark/run_benchmarks.jl [options]
#
# Options:
#   --set tax|cutest        problem set (default: tax)
#   --solvers a,b,...       subproblem solvers among ipopt, madnlp, knitro (default: ipopt,madnlp)
#   --problems A,B,...      restrict to these problems
#   --max-var N             CUTEst only: maximum number of variables (default: 100)
#   --max-con N             CUTEst only: maximum number of constraints (default: 100)
#   --max-time T            time limit in seconds per problem (default: 300)
#   --tag NAME              label for this run (default: current git branch)
#   --out-dir DIR           where to write results (default: benchmark/results)
#   --verbose               show NCL warnings
#
# Results go to <out-dir>/<tag>_<set>.csv, one row per (solver, problem), written as
# soon as each solve finishes so that an interrupted sweep keeps its results.

using AmplNLReader, CUTEst
using KNITRO, NLPModelsKnitro, MadNLP, NLPModelsIpopt
using NCL, NLPModels, SolverCore
using CSV, DataFrames, Logging, Printf

include("problems.jl")
include("solvers.jl")

function parse_args(args)
  opts = Dict{String, String}(
    "set" => "tax",
    "solvers" => "ipopt,madnlp",
    "problems" => "",
    "max-var" => "100",
    "max-con" => "100",
    "max-time" => "300",
    "tag" => "",
    "out-dir" => joinpath(@__DIR__, "results"),
  )
  verbose = false
  i = 1
  while i ≤ length(args)
    arg = args[i]
    if arg == "--verbose"
      verbose = true
    elseif startswith(arg, "--") && haskey(opts, arg[3:end]) && i < length(args)
      opts[arg[3:end]] = args[i + 1]
      i += 1
    else
      error("unrecognized argument $arg")
    end
    i += 1
  end
  return opts, verbose
end

split_list(s) = isempty(s) ? String[] : String.(split(s, ','))

git(args...) = try
  readchomp(Cmd(`git $args`; dir = @__DIR__))
catch
  "unknown"
end

function run_one(solve, name, constructor; max_time)
  nlp = constructor()
  row = (
    problem = name,
    nvar = get_nvar(nlp),
    ncon = get_ncon(nlp),
    status = "exception",
    sub_status = "",
    objective = NaN,
    residual = NaN,
    dual_feas = NaN,
    outer_iter = 0,
    inner_iter = 0,
    time = NaN,
    neval_obj = 0,
    neval_cons = 0,
    neval_hess = 0,
  )
  try
    stats = solve(nlp; max_time)
    r = stats.solver_specific[:residuals]
    row = merge(
      row,
      (
        status = string(stats.status),
        sub_status = string(stats.solver_specific[:sub_status]),
        objective = stats.objective,
        residual = isempty(r) ? 0.0 : maximum(abs, r),
        dual_feas = stats.dual_feas,
        outer_iter = stats.solver_specific[:outer_iter],
        inner_iter = stats.iter,
        time = stats.elapsed_time,
        neval_obj = neval_obj(nlp),
        neval_cons = neval_cons(nlp),
        neval_hess = neval_hess(nlp),
      ),
    )
  catch e
    e isa InterruptException && rethrow()
    @error "$name failed" exception = (e, catch_backtrace())
  finally
    finalize_model(nlp)
  end
  return row
end

function main(args)
  opts, verbose = parse_args(args)
  set = opts["set"]
  solvers = split_list(opts["solvers"])
  for s in solvers
    haskey(SOLVERS, s) || error("unknown solver $s; choose among $(join(keys(SOLVERS), ", "))")
  end
  max_time = parse(Float64, opts["max-time"])
  problems = problem_set(
    set;
    problems = split_list(opts["problems"]),
    max_var = parse(Int, opts["max-var"]),
    max_con = parse(Int, opts["max-con"]),
  )
  isempty(problems) && error("no problems selected")

  rev = git("rev-parse", "--short", "HEAD")
  dirty = isempty(git("status", "--porcelain", "--", "../src", "../ext")) ? "" : "-dirty"
  tag = isempty(opts["tag"]) ? git("rev-parse", "--abbrev-ref", "HEAD") : opts["tag"]
  tag = replace(tag, r"[^A-Za-z0-9_.-]" => "_")
  mkpath(opts["out-dir"])
  outfile = joinpath(opts["out-dir"], "$(tag)_$(set).csv")
  isfile(outfile) && rm(outfile)

  @info "benchmarking $(length(problems)) $set problems with $(join(solvers, ", ")) at $rev$dirty → $outfile"
  logger = verbose ? current_logger() : ConsoleLogger(stderr, Logging.Error)
  with_logger(logger) do
    for s in solvers
      solve = SOLVERS[s]
      # Compile before timing anything.
      run_one(solve, first(problems)...; max_time)
      for (k, (name, constructor)) in enumerate(problems)
        row = run_one(solve, name, constructor; max_time)
        row = merge((rev = rev * dirty, set = set, solver = s), row)
        CSV.write(outfile, [row]; append = isfile(outfile))
        @printf(
          stderr,
          "[%s %4d/%d] %-10s %-14s %5d %5d %9.2e %7.1e %7.2f\n",
          s,
          k,
          length(problems),
          name,
          row.status,
          row.outer_iter,
          row.inner_iter,
          row.objective,
          row.residual,
          row.time
        )
      end
    end
  end
  return outfile
end

main(ARGS)
