# Plot two benchmark runs produced by run_benchmarks.jl, in the style of the MadSuite slides.
#
#   julia --project=benchmark benchmark/plots.jl BASE.csv NEW.csv [OUTDIR]
#
# For each solver, write to OUTDIR (default benchmark/results/figures):
#   - <set>_<solver>_status.png: final status of each run;
#   - <set>_<solver>_profiles.png: performance profiles (BenchmarkProfiles) of the measures that
#     differ between the runs on problems both solve;
#   - <set>_<solver>_problems.png: inner iterations and wall time per problem, as grouped bars;
#   - <set>_<solver>_scaling.png: wall time against the number of variables.
# The last two are drawn for sets of ≤ 30 problems only.

using BenchmarkProfiles, CSV, DataFrames, OrderedCollections, Plots

default(framestyle = :box, gridalpha = 0.2, legend_background_color = :white, margin = 5Plots.mm)

const STATUSES = [  # label, statuses, color; best to worst
  ("solved", ["first_order"], "#0ca30c"),
  ("limit", ["max_iter", "max_time", "max_eval"], "#fab219"),
  ("infeasible", ["infeasible"], "#ec835a"),
  ("exception", ["exception"], "#d03b3b"),
]
const BARS = [:lightblue, :darkblue, :darkred]  # colors of the grouped bars
group(s) = something(findfirst(g -> s in g[2], STATUSES), length(STATUSES))
solved(df) = df.status .== "first_order"

function plot_status(stats, title)
  n = nrow(first(values(stats)))
  p = plot(; title, size = (760, 260), xlims = (0, n), ylims = (0.4, 2.6), yflip = true,
    yticks = (1:2, string.(keys(stats))), xlabel = "Number of problems", legend = :outerright)
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
  measures = [(c, n) for (c, n) in ((:time, "Wall time"), (:inner_iter, "Inner iterations"),
    (:outer_iter, "Outer iterations")) if !allequal(df[both, c] for df in values(stats))]
  ps = map(enumerate(measures)) do (k, (col, name))
    P = hcat([ifelse.(solved(df), Float64.(df[!, col]), Inf) for df in values(stats)]...)
    performance_profile(PlotsBackend(), P, string.(keys(stats)); title = name,
      legend = k == 1 ? :bottomright : false, linewidth = 1.5)
  end
  plot(ps...; layout = (1, length(ps)), size = (600 * length(ps), 480), plot_title = title,
    plot_titlefontsize = 11, left_margin = 8Plots.mm, bottom_margin = 8Plots.mm)
end

function plot_problems(stats, title)
  names = first(values(stats)).problem
  np, nr = length(names), length(stats)
  w = 0.8 / nr
  ps = map(((:inner_iter, "Inner iterations"), (:time, "Wall time (s)"))) do (col, name)
    p = plot(; ylabel = name, xticks = (1:np, names), xrotation = 30, xlims = (0.4, np + 0.6),
      ylims = (0, 1.4 * maximum(maximum(df[!, col]) for df in values(stats))),
      legend = col == :inner_iter ? :topleft : false, grid = :y)
    for (k, (label, df)) in enumerate(stats)
      xs = (1:np) .+ (k - (nr + 1) / 2) * w
      bar!(p, xs, df[!, col]; bar_width = w, color = BARS[k], linecolor = :black,
        fillalpha = ifelse.(solved(df), 1, 0.35), label = string(label))
      # Mark unsolved runs with their status above the bar.
      annotate!(p, [(x, v, text(" " * s, 7, :left, :bottom; rotation = 90))
                    for (x, v, s) in zip(xs, df[!, col], df.status) if s != "first_order"])
    end
    p
  end
  plot(ps...; layout = (1, 2), size = (1000, 420), plot_title = title, plot_titlefontsize = 11)
end

function plot_scaling(stats, title)
  p = plot(; title, xlabel = "Number of variables", ylabel = "Wall time (s)", xscale = :log10,
    yscale = :log10, legend = :topleft, size = (600, 450), titlefontsize = 11,
    xticks = (first(values(stats)).nvar, string.(first(values(stats)).nvar)))
  for (k, (label, df)) in enumerate(stats)
    plot!(p, df.nvar, df.time; color = k, marker = :circle, markersize = 4, label = string(label))
    # Unsolved runs are drawn as hollow markers.
    u = .!solved(df)
    any(u) && scatter!(p, df.nvar[u], df.time[u]; color = :white, markerstrokecolor = k,
      markersize = 4, label = "")
  end
  return p
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
    figs = ["status" => plot_status(stats, what), "profiles" => plot_profiles(stats, what)]
    if length(probs) ≤ 30
      push!(figs, "problems" => plot_problems(stats, what), "scaling" => plot_scaling(stats, what))
    end
    for (name, p) in figs
      savefig(p, joinpath(out, "$(set)_$(solver)_$name.png"))
    end
  end
end

main(ARGS...)
