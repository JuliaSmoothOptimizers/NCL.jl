# Plot two benchmark runs produced by run_benchmarks.jl with SolverBenchmark.
#
#   julia --project=benchmark benchmark/plots.jl BASE.csv NEW.csv [OUTDIR]
#
# For each solver, write to OUTDIR (default benchmark/results/figures) the performance profiles
# <set>_<solver>_profiles.png of the measures that differ between the runs on problems both solve.
# Problems a run does not solve count as failures in its profile.

using CSV, DataFrames, OrderedCollections, Plots, SolverBenchmark

const MEASURES = [:time => "Wall time", :inner_iter => "Inner iterations", :outer_iter => "Outer iterations"]

solved(df) = df.status .== "first_order"
cost(col) = df -> ifelse.(solved(df), Float64.(df[!, col]), Inf)

function plot_profiles(stats, title)
  both = .&(solved.(values(stats))...)
  measures = filter(m -> !allequal(df[both, m.first] for df in values(stats)), MEASURES)
  isempty(measures) && return nothing
  return profile_solvers(stats, cost.(first.(measures)), last.(measures);
    width = 500, height = 450, plot_title = title, plot_titlefontsize = 11,
    left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)
end

function main(basefile, newfile, out = joinpath(@__DIR__, "results", "figures"))
  runs = [CSV.read(f, DataFrame; stringtype = String) for f in (basefile, newfile)]
  labels = [Symbol(replace(basename(f), r"_[^_]*\.csv$" => "")) for f in (basefile, newfile)]
  set = runs[1].set[1]
  mkpath(out)
  for solver in intersect(unique.(getproperty.(runs, :solver))...)
    dfs = [df[df.solver .== solver, :] for df in runs]
    probs = sort(intersect(getproperty.(dfs, :problem)...))
    stats = OrderedDict(l => df[indexin(probs, df.problem), :] for (l, df) in zip(labels, dfs))
    p = plot_profiles(stats, "$set, NCL + $solver, $(length(probs)) problems")
    if p === nothing
      @warn "no measure differs between the runs" set solver
    else
      savefig(p, joinpath(out, "$(set)_$(solver)_profiles.png"))
    end
  end
end

main(ARGS...)
