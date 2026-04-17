# TODO: accept maximization problems

import NLPModels: increment!

export NCLModel

"""
    NCLModel(nlp)

Subtype of `AbstractNLPModel` designed to represent an NCL subproblem.
A general problem of the form

    minimize   f(x)
    over       x
    subject to lvar ≤ x ≤ uvar
               lcon ≤ c(x) ≤ ucon

is transformed into

    minimize   f(x) + λ'r + ρ ‖r‖²
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

# constructor
function NCLModel(
  nlp::AbstractNLPModel{T, S};
  resid::T = zero(T),
  resid_linear::Bool = true,
  ρ::T = one(T),
  y::S = fill!(similar(get_x0(nlp), resid_linear ? get_ncon(nlp) : get_nnln(nlp)), 1),
) where {T, S}
  if (get_ncon(nlp) == 0)
    @warn("input problem $(get_name(nlp)) is unconstrained, not generating NCL model")
    return nlp
  elseif ((get_nnln(nlp) == 0) && !resid_linear)
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
  x = view(xr, 1:(ncl.nx))
  r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
  obj_val = obj(ncl.nlp, x)
  get_minimize(ncl) || (obj_val *= -1)
  obj_res = ncl.y' * r + ncl.ρ * dot(r, r) / 2
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
  x = view(xr, 1:(ncl.nx))
  orig_gx = view(gx, 1:(ncl.nx))
  grad!(ncl.nlp, x, orig_gx)
  get_minimize(ncl) || (gx[1:(ncl.nx)] .*= -1)
  r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
  gx[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= ncl.ρ * r .+ ncl.y
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
  orig_nnzh = get_nnzh(ncl.nlp)
  orig_hrows = view(hrows, 1:orig_nnzh)
  orig_hcols = view(hcols, 1:orig_nnzh)
  hess_structure!(ncl.nlp, orig_hrows, orig_hcols)
  nnzh = get_nnzh(ncl)
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
  nnzh = get_nnzh(ncl)
  orig_nnzh = get_nnzh(ncl.nlp)
  x = view(xr, 1:(ncl.nx))
  orig_hvals = view(hvals, 1:orig_nnzh)
  hess_coord!(ncl.nlp, x, orig_hvals; obj_weight = obj_weight)
  # get_minimize(ncl) || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ * obj_weight
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
  nnzh = get_nnzh(ncl)
  orig_nnzh = get_nnzh(ncl.nlp)
  x = view(xr, 1:(ncl.nx))
  orig_hvals = view(hvals, 1:orig_nnzh)
  hess_coord!(ncl.nlp, x, y, orig_hvals; obj_weight = obj_weight)
  # get_minimize(ncl) || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ * obj_weight
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
  x = view(xr, 1:(ncl.nx))
  orig_hv = view(hv, 1:(ncl.nx))
  hprod!(ncl.nlp, x, view(v, 1:(ncl.nx)), orig_hv; obj_weight = obj_weight)
  # get_minimize(ncl) || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= 0
  else
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= obj_weight * ncl.ρ * v[(ncl.nx + 1):(ncl.nx + ncl.nr)]
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
  x = view(xr, 1:(ncl.nx))
  orig_hv = view(hv, 1:(ncl.nx))
  hprod!(ncl.nlp, x, y, view(v, 1:(ncl.nx)), orig_hv; obj_weight = obj_weight)
  # get_minimize(ncl) || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= 0
  else
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= obj_weight * ncl.ρ * v[(ncl.nx + 1):(ncl.nx + ncl.nr)]
  end
  # if get_minimize(ncl)
  #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
  # else
  #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= -ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
  # end
  return hv
end

function NLPModels.cons_lin!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  cx::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_nlin(ncl) cx
  increment!(ncl, :neval_cons_lin)
  x = view(xr, 1:(ncl.nx))
  cons_lin!(ncl.nlp, x, cx)
  if ncl.resid_linear
    r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
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
  x = view(xr, 1:(ncl.nx))
  cons_nln!(ncl.nlp, x, cx)
  r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
  if ncl.resid_linear
    cx .+= view(r, get_nln(ncl))
  else
    cx .+= r
  end
  return cx
end

function NLPModels.jac_lin_structure!(
  ncl::NCLModel{T, S, M},
  jrows::AbstractVector{<:Integer},
  jcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_lin_nnzj(ncl) jrows jcols
  increment!(ncl, :neval_jac_lin)
  orig_lin_nnzj = get_lin_nnzj(ncl.nlp)
  orig_jrows = view(jrows, 1:orig_lin_nnzj)
  orig_jcols = view(jcols, 1:orig_lin_nnzj)
  jac_lin_structure!(ncl.nlp, orig_jrows, orig_jcols)
  if ncl.resid_linear
    lin_nnzj = get_lin_nnzj(ncl)  # = orig_lin_nnzj + nlin
    nlin = get_nlin(ncl)
    jrows[(orig_lin_nnzj + 1):lin_nnzj] .= 1:nlin
    @. jcols[(orig_lin_nnzj + 1):lin_nnzj] = ncl.nx + (1:nlin)
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
  orig_nln_nnzj = get_nln_nnzj(ncl.nlp)
  orig_jrows = view(jrows, 1:orig_nln_nnzj)
  orig_jcols = view(jcols, 1:orig_nln_nnzj)
  jac_nln_structure!(ncl.nlp, orig_jrows, orig_jcols)
  nln_nnzj = get_nln_nnzj(ncl)
  nnln = get_nnln(ncl)
  jrows[(orig_nln_nnzj + 1):nln_nnzj] .= 1:nnln
  if ncl.resid_linear
    nlin = get_nlin(ncl)
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = ncl.nx + nlin + (1:nnln)
  else
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = ncl.nx + (1:nnln)
  end
  return jrows, jcols
end

function NLPModels.jac_lin_coord!(
  ncl::NCLModel{T, S, M},
  xr::AbstractVector,
  jvals::AbstractVector,
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  @lencheck get_lin_nnzj(ncl) jvals
  increment!(ncl, :neval_jac_lin)
  orig_lin_nnzj = get_lin_nnzj(ncl.nlp)
  orig_jvals = view(jvals, 1:orig_lin_nnzj)
  x = view(xr, 1:(ncl.nx))
  jac_lin_coord!(ncl.nlp, x, orig_jvals)
  if ncl.resid_linear
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
  orig_nln_nnzj = get_nln_nnzj(ncl.nlp)
  orig_jvals = view(jvals, 1:orig_nln_nnzj)
  x = view(xr, 1:(ncl.nx))
  jac_nln_coord!(ncl.nlp, x, orig_jvals)
  jvals[(orig_nln_nnzj + 1):get_nln_nnzj(ncl)] .= 1
  return jvals
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
  x = view(xr, 1:(ncl.nx))
  vx = view(v, 1:(ncl.nx))
  jprod_lin!(ncl.nlp, x, vx, Jv)
  vr = view(v, (ncl.nx + 1):(ncl.nx + ncl.nr))
  if ncl.resid_linear
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
  x = view(xr, 1:(ncl.nx))
  vx = view(v, 1:(ncl.nx))
  jprod_nln!(ncl.nlp, x, vx, Jv)
  vr = view(v, (ncl.nx + 1):(ncl.nx + ncl.nr))
  vr_nl = ncl.resid_linear ? view(vr, get_nln(ncl)) : vr
  Jv .+= vr_nl
  return Jv
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
  x = view(xr, 1:(ncl.nx))
  orig_Jtv = view(Jtv, 1:(ncl.nx))
  jtprod_lin!(ncl.nlp, x, v, orig_Jtv)
  if ncl.resid_linear
    Jtv[ncl.nx .+ get_lin(ncl)] .= v
    Jtv[ncl.nx .+ get_nln(ncl)] .= 0
  else
    Jtv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= 0
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
  x = view(xr, 1:(ncl.nx))
  orig_Jtv = view(Jtv, 1:(ncl.nx))
  jtprod_nln!(ncl.nlp, x, v, orig_Jtv)
  if ncl.resid_linear
    Jtv[ncl.nx .+ get_lin(ncl)] .= 0
    Jtv[ncl.nx .+ get_nln(ncl)] .= v
  else
    Jtv[(ncl.nx + 1):(ncl.nx + get_nnln(ncl))] .= v
  end
  return Jtv
end
