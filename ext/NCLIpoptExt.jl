module NCLIpoptExt

using NCL
using NLPModels
using NLPModelsIpopt

NCL._register_solver!(:ipopt)

function NCL._solve_ipopt(ncl::NLPModels.AbstractNLPModel; kwargs...)
  return NLPModelsIpopt.ipopt(ncl; kwargs...)
end

end
