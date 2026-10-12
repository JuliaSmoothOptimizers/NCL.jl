# Plot two benchmark runs produced by run_benchmarks.jl, in the style of the JSO tutorials.
#
#   julia --project=benchmark benchmark/plots.jl BASE.csv NEW.csv [OUTDIR]
#
# For each solver, write to OUTDIR (default benchmark/results/figures):
#   - <set>_<solver>_status.png: final status of each run;
#   - <set>_<solver>_profiles.png: performance profiles (SolverBenchmark.profile_solvers) of the
#     measures that differ between the runs on problems both solve;
#   - <set>_<solver>_problems.png: inner iterations and time per problem, for sets of ≤ 30 problems.

using CSV, DataFrames, OrderedCollections, Plots, SolverBenchmark

const STATUSES = [  # label, statuses, color; best to worst
  ("solved", ["first_order"], "#0ca30c"),
  ("limit", ["max_iter", "max_time", "max_eval"], "#fab219"),
  ("infeasible", ["infeasible"], "#ec835a"),
  ("exception", ["exception"], "#d03b3b"),
]
group(s) = something(findfirst(g -> s in g[2], STATUSES), length(STATUSES))
solved(df) = df.status .== "first_order"

function plot_status(stats, title)
  n = nrow(first(values(stats)))
  p = plot(; title, size = (760, 260), xlims = (0, n), ylims = (0.4, 2.6), yflip = true,
    yticks = (1:2, string.(keys(stats))), xlabel = "Number of problems", legend = :outerright,
    margin = 4Plots.mm)
  counts = [count(==(j), group.(df.status)) for df in values(stats), j in eachindex(STATUSES)]
  right = cumsum(counts; dims = 2)
  # Overlay cumulative bars, widest first, so each status shows as one segment.
  for j in reverse(findall(vec(any(counts .> 0; dims = 1))))
    bar!(p, 1:2, right[:, j]; orientation = :h, bar_width = 0.6, color = STATUSES[j][3],
      linecolor = :white, label = STATUSES[j][1])
    for k in 1:2
      counts[k, j] > n / 25 && annotate!(p, right[k, j] - counts[k, j] / 2, k, text(counts[k, j], 8))
    end
  end
  return p
end

function plot_profiles(stats, title)
  both = .&(solved.(values(stats))...)
  measures = [(c, n) for (c, n) in ((:time, "Time"), (:inner_iter, "Inner iterations"),
    (:outer_iter, "Outer iterations")) if !allequal(df[both, c] for df in values(stats))]
  costs = [df -> ifelse.(solved(df), df[!, c], Inf) for (c, _) in measures]
  profile_solvers(stats, costs, last.(measures); plot_title = title, plot_titlefontsize = 10,
    width = 450, margin = 6Plots.mm)
end

function plot_problems(stats, title)
  ps = map(((:inner_iter, "Inner iterations"), (:time, "Time (s)"))) do (col, name)
    v = [max.(df[!, col], 1.0e-2) for df in values(stats)]
    lo, hi = minimum(minimum.(v)) / 2, maximum(maximum.(v)) * 20
    np = length(v[1])
    p = plot(; title = name, xscale = :log10, xlims = (lo, hi), ylims = (0.4, np + 0.6),
      yticks = (1:np, first(values(stats)).problem), yflip = true,
      legend = col == :inner_iter ? :bottomright : false)
    for (k, (label, df)) in enumerate(stats)
      ys = axes(df, 1) .+ (k - 1.5) * 0.38
      bar!(p, ys, v[k]; orientation = :h, bar_width = 0.36, fillrange = lo, label = string(label),
        fillalpha = ifelse.(solved(df), 1, 0.3), color = k, linecolor = :white)
      annotate!(p, [(x * 1.3, y, text(s == "first_order" ? "" : s, 7, :left))
                    for (x, y, s) in zip(v[k], ys, df.status)])
    end
    p
  end
  plot(ps...; layout = (1, 2), size = (900, 200 + 45 * nrow(first(values(stats)))),
    plot_title = title, plot_titlefontsize = 12, plot_titlevspan = 0.08, margin = 4Plots.mm)
end

function main(basefile, newfile, out = joinpath(@__DIR__, "results", "figures"))
  runs = [CSV.read(f, DataFrame; stringtype = String) for f in (basefile, newfile)]
  labels = [Symbol(replace(basename(f), r"_[^_]*\.csv$" => "")) for f in (basefile, newfile)]
  set = runs[1].set[1]
  mkpath(out)
  for solver in intersect(unique.(getproperty.(runs, :solver))...)
    dfs = [sort(df[df.solver .== solver, :], [:nvar, :problem]) for df in runs]
    probs = intersect(getproperty.(dfs, :problem)...)
    stats = OrderedDict(l => df[indexin(probs, df.problem), :] for (l, df) in zip(labels, dfs))
    what = "$set, NCL + $solver, $(length(probs)) problems"
    figs = ["status" => plot_status(stats, what),
      "profiles" => plot_profiles(stats, what)]
    length(probs) ≤ 30 && push!(figs, "problems" => plot_problems(stats, what))
    for (name, p) in figs
      savefig(p, joinpath(out, "$(set)_$(solver)_$name.png"))
    end
  end
end

main(ARGS...)
