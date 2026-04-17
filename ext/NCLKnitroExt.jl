module NCLKnitroExt

using NCL
using NLPModels
using KNITRO
using NLPModelsKnitro

if KNITRO.has_knitro()
  NCL._register_solver!(:knitro)

  function NCL._solve_knitro(ncl::NLPModels.AbstractNLPModel; kwargs...)
    return NLPModelsKnitro.knitro(ncl; kwargs...)
  end
end

end
