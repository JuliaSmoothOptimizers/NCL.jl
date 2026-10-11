# Compare two benchmark runs produced by run_benchmarks.jl, e.g. main against a branch.
#
#   julia --project=benchmark benchmark/compare.jl BASE.csv NEW.csv [--rtol 1e-4]
#
# For each solver present in both runs, report:
#   - how many problems each run solves (status first_order);
#   - problems whose status changed;
#   - problems solved by both whose objective differs by more than rtol (relative);
#   - the change in inner iterations, outer iterations and time on problems solved by both,
#     as a shifted geometric mean ratio NEW / BASE (< 1 means NEW is better).

using CSV, DataFrames, Printf, Statistics

const SOLVED = "first_order"

# Shifted geometric mean, the usual summary for iteration counts and times.
sgm(x; shift = 1) = exp(mean(log.(x .+ shift))) - shift

function parse_args(args)
  files = String[]
  rtol = 1.0e-4
  i = 1
  while i ≤ length(args)
    if args[i] == "--rtol"
      rtol = parse(Float64, args[i + 1])
      i += 1
    else
      push!(files, args[i])
    end
    i += 1
  end
  length(files) == 2 || error("usage: compare.jl BASE.csv NEW.csv [--rtol 1e-4]")
  return files..., rtol
end

function report(io, solver, base, new, rtol)
  df = innerjoin(base, new; on = :problem, renamecols = "_base" => "_new")
  println(io, "\n## $solver: $(nrow(df)) problems in both runs\n")

  nb = count(==(SOLVED), df.status_base)
  nn = count(==(SOLVED), df.status_new)
  @printf(io, "solved: %d → %d\n", nb, nn)

  changed = df[df.status_base .!= df.status_new, :]
  if nrow(changed) > 0
    println(io, "\nstatus changed:")
    for r in eachrow(changed)
      @printf(io, "  %-12s %-14s → %s\n", r.problem, r.status_base, r.status_new)
    end
  end

  both = df[(df.status_base .== SOLVED) .& (df.status_new .== SOLVED), :]
  reldiff = @. abs(both.objective_new - both.objective_base) / max(1, abs(both.objective_base))
  objdiff = both[reldiff .> rtol, :]
  if nrow(objdiff) > 0
    println(io, "\nobjective differs (rtol = $rtol) although both solved:")
    for r in eachrow(objdiff)
      @printf(io, "  %-12s %15.8e → %15.8e\n", r.problem, r.objective_base, r.objective_new)
    end
  end

  if nrow(both) > 0
    println(io, "\non the $(nrow(both)) problems solved by both (NEW / BASE, < 1 is better):")
    for (col, shift) in ((:inner_iter, 1), (:outer_iter, 1), (:time, 1))
      b = df_col(both, col, "_base")
      n = df_col(both, col, "_new")
      @printf(
        io,
        "  %-10s  sgm %8.2f → %8.2f  (ratio %.2f)   fewer on %d, more on %d\n",
        col,
        sgm(b; shift),
        sgm(n; shift),
        sgm(n; shift) / sgm(b; shift),
        count(n .< b),
        count(n .> b)
      )
    end
  end

  println(io, "\nper problem:")
  @printf(
    io,
    "  %-12s %-14s %-14s %7s %7s %9s %9s\n",
    "problem",
    "status base",
    "status new",
    "it base",
    "it new",
    "t base",
    "t new"
  )
  for r in eachrow(sort(df, :problem))
    @printf(
      io,
      "  %-12s %-14s %-14s %7d %7d %9.2f %9.2f\n",
      r.problem,
      r.status_base,
      r.status_new,
      r.inner_iter_base,
      r.inner_iter_new,
      r.time_base,
      r.time_new
    )
  end
end

df_col(df, col, suffix) = Float64.(df[!, Symbol(string(col) * suffix)])

function main(args)
  basefile, newfile, rtol = parse_args(args)
  base = CSV.read(basefile, DataFrame; stringtype = String)
  new = CSV.read(newfile, DataFrame; stringtype = String)
  println("BASE: $basefile (rev $(join(unique(base.rev), ", ")))")
  println("NEW:  $newfile (rev $(join(unique(new.rev), ", ")))")
  for solver in sort(intersect(unique(base.solver), unique(new.solver)))
    report(stdout, solver, base[base.solver .== solver, :], new[new.solver .== solver, :], rtol)
  end
end

main(ARGS)
