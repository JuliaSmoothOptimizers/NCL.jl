module NCLIpoptExt

using NCL
using NLPModels
using NLPModelsIpopt

function __init__()
  @info "Registering NCL solver IPOPT"
  NCL._register_solver!(:ipopt)
end

const ipopt_fixed_options = Dict(
  :sb => "yes",  # options that are always used
  :print_level => 0,
  :max_iter => 100,
)

function NCL._solve_ipopt(ncl::NLPModels.AbstractNLPModel; kwargs...)
  return NLPModelsIpopt.ipopt(ncl; ipopt_fixed_options..., kwargs...)
end

end
