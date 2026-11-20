# Problem sets for benchmarking NCL.
#
# Each set is a vector of `name => constructor` pairs, where `constructor()` returns
# an `AbstractNLPModel`. Models are built lazily so that only one CUTEst problem is
# decoded and loaded at a time.

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

const TAX_PROBLEMS = ["tax1D", "tax2D", "pTax3D", "pTax4D", "pTax5D"]

tax_problems(names = TAX_PROBLEMS) =
  [name => () -> AmplModel(joinpath(DATA_DIR, "$name.nl")) for name in names]

"""
    cutest_problems(; max_var = 100, max_con = 100, kwargs...)

Constrained CUTEst problems with at most `max_var` variables and `max_con` constraints,
sorted by name. Other keyword arguments are passed to `CUTEst.select_sif_problems`.
"""
function cutest_problems(; max_var = 100, max_con = 100, kwargs...)
  names = sort(select_sif_problems(; min_con = 1, max_var, max_con, kwargs...))
  return [name => () -> CUTEstModel(name) for name in names]
end

function problem_set(set::AbstractString; problems = String[], max_var = 100, max_con = 100)
  if set == "tax"
    pbs = tax_problems()
  elseif set == "cutest"
    pbs = isempty(problems) ? cutest_problems(; max_var, max_con) :
          [name => () -> CUTEstModel(name) for name in problems]
  else
    error("unknown problem set $set; use \"tax\" or \"cutest\"")
  end
  if set == "tax" && !isempty(problems)
    pbs = filter(p -> first(p) in problems, pbs)
  end
  return pbs
end

finalize_model(::AbstractNLPModel) = nothing
finalize_model(nlp::CUTEstModel) = finalize(nlp)
