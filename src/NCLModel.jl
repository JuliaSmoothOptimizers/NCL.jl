# TODO: accept maximization problems

import NLPModels: increment!

export NCLModel
export get_nlp, get_nx, get_nr
export set_penalty_parameter!,
  get_penalty_parameter, get_multipliers, add_to_multipliers!, get_resid_linear

"""
    NCLModel(nlp)

Subtype of `AbstractNLPModel` designed to represent an NCL subproblem.
A general problem of the form

    minimize   f(x)
    over       x
    subject to lvar ≤ x ≤ uvar
               lcon ≤ c(x) ≤ ucon

is transformed into

    minimize   f(x) + λ'r + 1/2 ρ ‖r‖²
    over       x, r
    subject to lvar ≤ x ≤ uvar
               lcon ≤ c(x) + r ≤ ucon

where λ is a vector of Lagrange multiplier estimates and ρ > 0 is a penalty parameter.

### Input arguments

* `nlp::AbstractNLPModel`  the original problem

### Keyword arguments

* `resid::Float64`  the initial residual value (default 0)
* `resid_linear::Bool`  whether or not residuals are added to linear constraints
* `ρ::Float64`  initial penalty parameter
* `y::AbstractVector{Float64}`  initial Lagrange multiplier estimates

### Return value

* `ncl::NCLModel`  the transformed model.
"""
mutable struct NCLModel{T, S, M} <: AbstractNLPModel{T, S} where {M <: AbstractNLPModel{T, S}}
  nlp::M
  nx::Int  # number of variables in nlp
  nr::Int  # number of residuals added in the NCL problem (get_ncon(nlp) if resid_linear, else get_nnln(nlp))
  resid_linear::Bool

  meta::NLPModelMeta{T, S}
  counters::Counters

  y::S
  ρ::T # penalty parameter
end

NLPModels.reset!(ncl::NCLModel) = begin
  NLPModels.reset!(ncl.nlp)
  NLPModels.reset!(ncl.counters)
  ncl
end

get_nlp(ncl::NCLModel) = ncl.nlp
get_nx(ncl::NCLModel) = ncl.nx
get_nr(ncl::NCLModel) = ncl.nr
get_penalty_parameter(ncl::NCLModel) = ncl.ρ
set_penalty_parameter!(ncl::NCLModel{T, S, M}, ρ::T) where {T, S, M} = ncl.ρ = max(ρ, zero(T))
get_multipliers(ncl::NCLModel) = ncl.y
add_to_multipliers!(ncl::NCLModel{T, S, M}, α::T, v::S) where {T, S, M} = ncl.y .+= α .* v
get_resid_linear(ncl::NCLModel) = ncl.resid_linear

# constructor
function NCLModel(
  nlp::AbstractNLPModel{T, S};
  resid::T = zero(T),
  resid_linear::Bool = true,
  ρ::T = one(T),
  y::S = fill!(similar(get_x0(nlp), resid_linear ? get_ncon(nlp) : get_nnln(nlp)), 1),
) where {T, S}
  if unconstrained(nlp) || bound_constrained(nlp)
    @warn(
      "input problem $(get_name(nlp)) is unconstrained or bound constrained, not generating NCL model"
    )
    return nlp
  elseif linearly_constrained(nlp) && !resid_linear
    @warn(
      "input problem $(get_name(nlp)) is linearly constrained and `resid_linear` is `false`, not generating NCL model"
    )
    return nlp
  end

  # number of residuals
  nr = resid_linear ? get_ncon(nlp) : get_nnln(nlp)

  # construct meta
  nx = get_nvar(nlp)
  nvar = nx + nr
  nlin = get_nlin(nlp)
  nnln = get_nnln(nlp)
  lin_nnzj = get_lin_nnzj(nlp) + (resid_linear ? nlin : 0)
  nln_nnzj = get_nln_nnzj(nlp) + nnln
  meta = NLPModelMeta{T, S}(
    nvar,
    lvar = vcat(get_lvar(nlp), fill!(similar(get_x0(nlp), nr), -Inf)),  # no bounds on residuals
    uvar = vcat(get_uvar(nlp), fill!(similar(get_x0(nlp), nr), Inf)),
    x0 = vcat(get_x0(nlp), fill!(similar(get_x0(nlp), nr), resid)),
    y0 = get_y0(nlp),
    name = "NCL-" * get_name(nlp),
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    nnzj = lin_nnzj + nln_nnzj,
    lin = get_lin(nlp),  # nln is automatically computed
    nnzh = get_nnzh(nlp) + nr,
    ncon = get_ncon(nlp),
    lcon = get_lcon(nlp),
    ucon = get_ucon(nlp),
    minimize = true,  # get_minimize(nlp)
    islp = false,
    sparse_jacobian = get_sparse_jacobian(nlp),
    sparse_hessian = get_sparse_hessian(nlp),
    grad_available = get_grad_available(nlp),
    jac_available = get_jac_available(nlp),
    hess_available = get_hess_available(nlp),
    jprod_available = get_jprod_available(nlp),
    jtprod_available = get_jtprod_available(nlp),
    hprod_available = get_hprod_available(nlp),
  )

  get_minimize(nlp) || error("only minimization problems are currently supported")
  return NCLModel{T, S, typeof(nlp)}(nlp, nx, nr, resid_linear, meta, Counters(), y, ρ)
end

function NLPModels.obj(ncl::NCLModel{T, S, M}, xr::S) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  increment!(ncl, :neval_obj)

  nlp = get_nlp(ncl)
  n = get_nvar(ncl)
  nx = get_nx(ncl)
  y = get_multipliers(ncl)
  ρ = get_penalty_parameter(ncl)

  x = view(xr, 1:nx)
  r = view(xr, (nx + 1):n)

  obj_val = obj(nlp, x)
  get_minimize(ncl) || (obj_val *= -1)
  obj_res = y' * r + ρ * dot(r, r) / 2
  # get_minimize(ncl) || (obj_res *= -1)
  return obj_val + obj_res
end

function NLPModels.grad!(
  ncl::NCLModel{T, S, M},
  xr::S,
  gx::S,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr gx
  increment!(ncl, :neval_grad)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)
  y = get_multipliers(ncl)
  ρ = get_penalty_parameter(ncl)

  x = view(xr, 1:nx)
  r = view(xr, (nx + 1):n)
  orig_gx = view(gx, 1:nx)

  grad!(nlp, x, orig_gx)
  get_minimize(ncl) || (gx[1:nx] .*= -1)
  gx[(nx + 1):n] .= ρ * r .+ y
  # get_minimize(ncl) || (gx[ncl.nx + 1 : ncl.nx + ncl.nr] .*= -1)
  return gx
end

function NLPModels.hess_structure!(
  ncl::NCLModel{T, S, M},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nnzh(ncl) hrows hcols
  increment!(ncl, :neval_hess)

  nlp = get_nlp(ncl)
  orig_nnzh = get_nnzh(nlp)
  nnzh = get_nnzh(ncl)

  orig_hrows = view(hrows, 1:orig_nnzh)
  orig_hcols = view(hcols, 1:orig_nnzh)

  hess_structure!(nlp, orig_hrows, orig_hcols)
  hrows[(orig_nnzh + 1):nnzh] .= (ncl.nx + 1):(get_nvar(ncl))
  hcols[(orig_nnzh + 1):nnzh] .= (ncl.nx + 1):(get_nvar(ncl))
  return (hrows, hcols)
end

function NLPModels.hess_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  hvals::AbstractVector;
  obj_weight::T = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nnzh(ncl) hvals
  increment!(ncl, :neval_hess)

  nlp = get_nlp(ncl)
  nnzh = get_nnzh(ncl)
  nx = get_nx(ncl)
  orig_nnzh = get_nnzh(nlp)
  ρ = get_penalty_parameter(ncl)

  x = view(xr, 1:nx)
  orig_hvals = view(hvals, 1:orig_nnzh)

  hess_coord!(nlp, x, orig_hvals; obj_weight = obj_weight)
  # get_minimize(ncl) || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ρ * obj_weight
  # if get_minimize(ncl)
  # hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ
  # else
  #   hvals[orig_nnzh + 1 : nnzh] .= -ncl.ρ
  # end
  return hvals
end

function NLPModels.hess_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  y::AbstractVector,
  hvals::AbstractVector;
  obj_weight::T = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_ncon(ncl) y
  @lencheck get_nnzh(ncl) hvals
  increment!(ncl, :neval_hess)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  nnzh = get_nnzh(ncl)
  orig_nnzh = get_nnzh(nlp)
  ρ = get_penalty_parameter(ncl)

  x = view(xr, 1:nx)
  orig_hvals = view(hvals, 1:orig_nnzh)

  hess_coord!(nlp, x, y, orig_hvals; obj_weight = obj_weight)
  # get_minimize(ncl) || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ρ * obj_weight
  # if get_minimize(ncl)
  # hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ
  # else
  #   hvals[orig_nnzh + 1 : nnzh] .= -ncl.ρ
  # end
  return hvals
end

function NLPModels.hprod!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::T = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr v hv
  increment!(ncl, :neval_hprod)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  orig_hv = view(hv, 1:nx)

  hprod!(nlp, x, view(v, 1:nx), orig_hv; obj_weight = obj_weight)
  # get_minimize(ncl) || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):n] .= 0
  else
    ρ = get_penalty_parameter(ncl)
    hv[(nx + 1):n] .= obj_weight * ρ * v[(nx + 1):n]
  end
  # if get_minimize(ncl)
  # hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= ncl.ρ * v[(ncl.nx + 1):(ncl.nx + ncl.nr)]
  # else
  #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= -ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
  # end
  return hv
end

function NLPModels.hprod!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::T = one(T),
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr v hv
  @lencheck get_ncon(ncl) y
  increment!(ncl, :neval_hprod)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  orig_hv = view(hv, 1:nx)

  hprod!(nlp, x, y, view(v, 1:nx), orig_hv; obj_weight = obj_weight)
  # get_minimize(ncl) || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):n] .= 0
  else
    ρ = get_penalty_parameter(ncl)
    hv[(ncl.nx + 1):n] .= obj_weight * ρ * v[(nx + 1):n]
  end
  # if get_minimize(ncl)
  #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
  # else
  #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= -ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
  # end
  return hv
end

# Implement cons! for models that do not support the linear/nonlinear API.
function NLPModels.cons!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  cx::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_ncon(ncl) cx
  increment!(ncl, :neval_cons)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  r = view(xr, (nx + 1):n)

  cons!(nlp, x, cx)
  if get_resid_linear(ncl)
    cx .+= r
  else
    nln = get_nln(ncl)
    cx[nln] .+= r
  end
  return cx
end

function NLPModels.cons_lin!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  cx::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nlin(ncl) cx
  increment!(ncl, :neval_cons_lin)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)

  cons_lin!(nlp, x, cx)
  if get_resid_linear(ncl)
    r = view(xr, (nx + 1):n)
    cx .+= view(r, get_lin(ncl))
  end
  return cx
end

function NLPModels.cons_nln!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  cx::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nnln(ncl) cx
  increment!(ncl, :neval_cons_nln)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  r = view(xr, (nx + 1):n)

  cons_nln!(nlp, x, cx)
  if get_resid_linear(ncl)
    cx .+= view(r, get_nln(ncl))
  else
    cx .+= r
  end
  return cx
end

# Implement jac_structure! for models that do not support the linear/nonlinear API.
function NLPModels.jac_structure!(
  ncl::NCLModel{T, S, M},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nnzj(ncl) jrows jcols
  increment!(ncl, :neval_jac)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)
  orig_nnzj = get_nnzj(nlp)
  nnzj = get_nnzj(ncl)

  orig_jrows = view(jrows, 1:orig_nnzj)
  orig_jcols = view(jcols, 1:orig_nnzj)

  jac_structure!(nlp, orig_jrows, orig_jcols)
  jrows[(orig_nnzj + 1):nnzj] .= get_resid_linear(ncl) ? (1:get_ncon(ncl)) : get_nln(ncl)
  jcols[(orig_nnzj + 1):nnzj] .= (nx + 1):n
  return jrows, jcols
end

function NLPModels.jac_lin_structure!(
  ncl::NCLModel{T, S, M},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_lin_nnzj(ncl) jrows jcols
  increment!(ncl, :neval_jac_lin)

  nlp = get_nlp(ncl)
  orig_lin_nnzj = get_lin_nnzj(nlp)

  orig_jrows = view(jrows, 1:orig_lin_nnzj)
  orig_jcols = view(jcols, 1:orig_lin_nnzj)

  jac_lin_structure!(nlp, orig_jrows, orig_jcols)
  if get_resid_linear(ncl)
    nx = get_nx(ncl)
    lin_nnzj = get_lin_nnzj(ncl)  # = orig_lin_nnzj + nlin
    nlin = get_nlin(ncl)
    jrows[(orig_lin_nnzj + 1):lin_nnzj] .= 1:nlin
    @. jcols[(orig_lin_nnzj + 1):lin_nnzj] = nx + (1:nlin)
  end
  return jrows, jcols
end

function NLPModels.jac_nln_structure!(
  ncl::NCLModel{T, S, M},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nln_nnzj(ncl) jrows jcols
  increment!(ncl, :neval_jac_nln)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  orig_nln_nnzj = get_nln_nnzj(nlp)
  nln_nnzj = get_nln_nnzj(ncl)
  nnln = get_nnln(ncl)

  orig_jrows = view(jrows, 1:orig_nln_nnzj)
  orig_jcols = view(jcols, 1:orig_nln_nnzj)

  jac_nln_structure!(nlp, orig_jrows, orig_jcols)
  jrows[(orig_nln_nnzj + 1):nln_nnzj] .= 1:nnln
  if get_resid_linear(ncl)
    nlin = get_nlin(ncl)
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = nx + nlin + (1:nnln)
  else
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = nx + (1:nnln)
  end
  return jrows, jcols
end

function NLPModels.jac_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  jvals::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nnzj(ncl) jvals
  increment!(ncl, :neval_jac)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)

  orig_nnzj = get_nnzj(nlp)
  orig_jvals = view(jvals, 1:orig_nnzj)
  x = view(xr, 1:nx)

  jac_coord!(nlp, x, orig_jvals)
  jvals[(orig_nnzj + 1):get_nnzj(ncl)] .= 1
  return jvals
end

function NLPModels.jac_lin_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  jvals::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_lin_nnzj(ncl) jvals
  increment!(ncl, :neval_jac_lin)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)

  orig_lin_nnzj = get_lin_nnzj(nlp)
  orig_jvals = view(jvals, 1:orig_lin_nnzj)
  x = view(xr, 1:nx)

  jac_lin_coord!(nlp, x, orig_jvals)
  if get_resid_linear(ncl)
    jvals[(orig_lin_nnzj + 1):get_lin_nnzj(ncl)] .= 1
  end
  return jvals
end

function NLPModels.jac_nln_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  jvals::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nln_nnzj(ncl) jvals
  increment!(ncl, :neval_jac_nln)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)

  orig_nln_nnzj = get_nln_nnzj(nlp)
  orig_jvals = view(jvals, 1:orig_nln_nnzj)
  x = view(xr, 1:nx)

  jac_nln_coord!(nlp, x, orig_jvals)
  jvals[(orig_nln_nnzj + 1):get_nln_nnzj(ncl)] .= 1
  return jvals
end

function NLPModels.jprod!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr v
  @lencheck get_ncon(ncl) Jv
  increment!(ncl, :neval_jprod)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  vx = view(v, 1:nx)
  vr = view(v, (nx + 1):n)

  jprod!(nlp, x, vx, Jv)
  if get_resid_linear(ncl)
    Jv .+= vr
  else
    Jv[get_nln(ncl)] .+= vr
  end
  return Jv
end

function NLPModels.jprod_lin!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr v
  @lencheck get_nlin(ncl) Jv
  increment!(ncl, :neval_jprod_lin)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  vx = view(v, 1:nx)
  vr = view(v, (nx + 1):n)

  jprod_lin!(nlp, x, vx, Jv)
  if get_resid_linear(ncl)
    vr_lin = view(vr, get_lin(ncl))
    Jv .+= vr_lin
  end
  return Jv
end

function NLPModels.jprod_nln!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr v
  @lencheck get_nnln(ncl) Jv
  increment!(ncl, :neval_jprod_nln)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  vx = view(v, 1:nx)
  vr = view(v, (nx + 1):n)

  jprod_nln!(nlp, x, vx, Jv)
  vr_nl = get_resid_linear(ncl) ? view(vr, get_nln(ncl)) : vr
  Jv .+= vr_nl
  return Jv
end

function NLPModels.jtprod!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr Jtv
  @lencheck get_ncon(ncl) v
  increment!(ncl, :neval_jtprod)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  orig_Jtv = view(Jtv, 1:nx)

  jtprod!(nlp, x, v, orig_Jtv)
  Jtv[(nx + 1):n] .= get_resid_linear(ncl) ? v : view(v, get_nln(ncl))
  return Jtv
end

function NLPModels.jtprod_lin!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr Jtv
  @lencheck get_nlin(ncl) v
  increment!(ncl, :neval_jtprod_lin)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)
  n = get_nvar(ncl)

  x = view(xr, 1:nx)
  orig_Jtv = view(Jtv, 1:nx)

  jtprod_lin!(nlp, x, v, orig_Jtv)
  if get_resid_linear(ncl)
    Jtv[nx .+ get_lin(ncl)] .= v
    Jtv[nx .+ get_nln(ncl)] .= 0
  else
    Jtv[(nx + 1):n] .= 0
  end
  return Jtv
end

function NLPModels.jtprod_nln!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  v::AbstractVector,    # v has length nnln
  Jtv::AbstractVector,  # Jtv has length nvar = nx + nr
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr Jtv
  @lencheck get_nnln(ncl) v
  increment!(ncl, :neval_jtprod_nln)

  nlp = get_nlp(ncl)
  nx = get_nx(ncl)

  x = view(xr, 1:nx)
  orig_Jtv = view(Jtv, 1:nx)

  jtprod_nln!(nlp, x, v, orig_Jtv)
  if get_resid_linear(ncl)
    Jtv[nx .+ get_lin(ncl)] .= 0
    Jtv[nx .+ get_nln(ncl)] .= v
  else
    Jtv[(nx + 1):(nx + get_nnln(ncl))] .= v
  end
  return Jtv
end
