module NCLKnitroExt

using NCL
using NLPModels
using KNITRO
using NLPModelsKnitro

function __init__()
  if KNITRO.has_knitro()
    @info "Registering NCL solver KNITRO"
    NCL._register_solver!(:knitro)
  else
    @warn "KNITRO is not available"
  end
end

const knitro_fixed_options =
  Dict(:algorithm => 1, :bar_directinterval => 0, :bar_initpt => 2, :outlev => 0, :maxit => 100)

function NCL._solve_knitro(ncl::NLPModels.AbstractNLPModel; kwargs...)
  if KNITRO.has_knitro()
    return NLPModelsKnitro.knitro(ncl; knitro_fixed_options..., kwargs...)
  end

  error(
    "Knitro support is loaded, but Knitro is not available/configured. " *
    "Please check that the Knitro binary is installed and that its license is configured.",
  )
end

end
