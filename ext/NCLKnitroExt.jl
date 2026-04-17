module NCLKnitroExt

using NCL
using NLPModels
using KNITRO
using NLPModelsKnitro

if KNITRO.has_knitro()
  NCL._register_solver!(:knitro)
end

function NCL._solve_knitro(ncl::NLPModels.AbstractNLPModel; kwargs...)
  if KNITRO.has_knitro()
    return NLPModelsKnitro.knitro(ncl; kwargs...)
  end

  error(
    "Knitro support is loaded, but Knitro is not available/configured. " *
    "Please check that the Knitro binary is installed and that its license is configured."
  )
end

end
