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
  nr::Int  # number of residuals in nlp problem (nr = length(nlp.meta.nln))
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
  y::S = fill!(similar(nlp.meta.x0, resid_linear ? nlp.meta.ncon : nlp.meta.nnln), 1),
) where {T, S}
  if (nlp.meta.ncon == 0)
    @warn("input problem $(nlp.meta.name) is unconstrained, not generating NCL model")
    return nlp
  elseif ((nlp.meta.nnln == 0) & !resid_linear)
    @warn(
      "input problem $(nlp.meta.name) is linearly constrained and `resid_linear` is `false`, not generating NCL model"
    )
    return nlp
  end

  # number of residuals
  nr = resid_linear ? nlp.meta.ncon : nlp.meta.nnln

  # construct meta
  nx = nlp.meta.nvar
  nvar = nx + nr
  nlin = nlp.meta.nlin
  nnln = nlp.meta.nnln
  lin_nnzj = nlp.meta.lin_nnzj + (resid_linear ? nlin : 0)
  nln_nnzj = nlp.meta.nln_nnzj + nnln
  meta = NLPModelMeta{T, S}(
    nvar,
    lvar = vcat(nlp.meta.lvar, fill!(similar(nlp.meta.x0, nr), -Inf)),  # no bounds on residuals
    uvar = vcat(nlp.meta.uvar, fill!(similar(nlp.meta.x0, nr), Inf)),
    x0 = vcat(nlp.meta.x0, fill!(similar(nlp.meta.x0, nr), resid)),
    y0 = nlp.meta.y0,
    name = "NCL-" * nlp.meta.name,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    nnzj = lin_nnzj + nln_nnzj,
    lin = nlp.meta.lin,  # nln is automatically computed
    nnzh = nlp.meta.nnzh + nr,
    ncon = nlp.meta.ncon,
    lcon = nlp.meta.lcon,
    ucon = nlp.meta.ucon,
    minimize = true,  # nlp.meta.minimize,
    islp = false,
    sparse_jacobian = nlp.meta.sparse_jacobian,
    sparse_hessian = nlp.meta.sparse_hessian,
    grad_available = nlp.meta.grad_available,
    jac_available = nlp.meta.jac_available,
    hess_available = nlp.meta.hess_available,
    jprod_available = nlp.meta.jprod_available,
    jtprod_available = nlp.meta.jtprod_available,
    hprod_available = nlp.meta.hprod_available,
  )

  nlp.meta.minimize || error("only minimization problems are currently supported")
  return NCLModel{T, S, typeof(nlp)}(nlp, nx, nr, resid_linear, meta, Counters(), y, ρ)
end

function NLPModels.obj(ncl::NCLModel{T, S, M}, xr::S) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nvar(ncl) xr
  increment!(ncl, :neval_obj)
  x = view(xr, 1:(ncl.nx))
  r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
  obj_val = obj(ncl.nlp, x)
  ncl.nlp.meta.minimize || (obj_val *= -1)
  obj_res = ncl.y' * r + ncl.ρ * dot(r, r) / 2
  # ncl.meta.minimize || (obj_res *= -1)
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
  ncl.nlp.meta.minimize || (gx[1:(ncl.nx)] .*= -1)
  r = view(xr, (ncl.nx + 1):(ncl.nx + ncl.nr))
  gx[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= ncl.ρ * r .+ ncl.y
  # ncl.meta.minimize || (gx[ncl.nx + 1 : ncl.nx + ncl.nr] .*= -1)
  return gx
end

function NLPModels.hess_structure!(
  ncl::NCLModel{T, S, M},
  hrows::AbstractVector{<:Integer},
  hcols::AbstractVector{<:Integer},
) where {T, S, M <: AbstractNLPModel{T, S}}
  @lencheck get_nnzh(ncl) hrows hcols
  increment!(ncl, :neval_hess)
  orig_nnzh = ncl.nlp.meta.nnzh
  orig_hrows = view(hrows, 1:orig_nnzh)
  orig_hcols = view(hcols, 1:orig_nnzh)
  hess_structure!(ncl.nlp, orig_hrows, orig_hcols)
  nnzh = ncl.meta.nnzh
  hrows[(orig_nnzh + 1):nnzh] .= (ncl.nx + 1):(ncl.meta.nvar)
  hcols[(orig_nnzh + 1):nnzh] .= (ncl.nx + 1):(ncl.meta.nvar)
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
  nnzh = ncl.meta.nnzh
  orig_nnzh = ncl.nlp.meta.nnzh
  x = view(xr, 1:(ncl.nx))
  orig_hvals = view(hvals, 1:orig_nnzh)
  hess_coord!(ncl.nlp, x, orig_hvals; obj_weight = obj_weight)
  # ncl.nlp.meta.minimize || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ * obj_weight
  # if ncl.meta.minimize
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
  nnzh = ncl.meta.nnzh
  orig_nnzh = ncl.nlp.meta.nnzh
  x = view(xr, 1:(ncl.nx))
  orig_hvals = view(hvals, 1:orig_nnzh)
  hess_coord!(ncl.nlp, x, y, orig_hvals; obj_weight = obj_weight)
  # ncl.nlp.meta.minimize || (hvals[1:orig_nnzh] .*= -1)
  hvals[(orig_nnzh + 1):nnzh] .= ncl.ρ * obj_weight
  # if ncl.meta.minimize
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
  # ncl.nlp.meta.minimize || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= 0
  else
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= obj_weight * ncl.ρ * v[(ncl.nx + 1):(ncl.nx + ncl.nr)]
  end
  # if ncl.meta.minimize
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
  # ncl.nlp.meta.minimize || (orig_hv .*= -1)
  if obj_weight == zero(T)
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= 0
  else
    hv[(ncl.nx + 1):(ncl.nx + ncl.nr)] .= obj_weight * ncl.ρ * v[(ncl.nx + 1):(ncl.nx + ncl.nr)]
  end
  # if ncl.meta.minimize
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
    cx .+= view(r, ncl.meta.lin)
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
    cx .+= view(r, ncl.meta.nln)
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
  orig_lin_nnzj = ncl.nlp.meta.lin_nnzj
  orig_jrows = view(jrows, 1:orig_lin_nnzj)
  orig_jcols = view(jcols, 1:orig_lin_nnzj)
  jac_lin_structure!(ncl.nlp, orig_jrows, orig_jcols)
  if ncl.resid_linear
    lin_nnzj = ncl.meta.lin_nnzj  # = orig_lin_nnzj + nlin
    jrows[(orig_lin_nnzj + 1):lin_nnzj] .= 1:ncl.meta.nlin
    @. jcols[(orig_lin_nnzj + 1):lin_nnzj] = ncl.nx + (1:ncl.meta.nlin)
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
  orig_nln_nnzj = ncl.nlp.meta.nln_nnzj
  orig_jrows = view(jrows, 1:orig_nln_nnzj)
  orig_jcols = view(jcols, 1:orig_nln_nnzj)
  jac_nln_structure!(ncl.nlp, orig_jrows, orig_jcols)
  nln_nnzj = ncl.meta.nln_nnzj
  jrows[(orig_nln_nnzj + 1):nln_nnzj] .= 1:ncl.meta.nnln
  if ncl.resid_linear
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = ncl.nx + ncl.meta.nlin + (1:ncl.meta.nnln)
  else
    @. jcols[(orig_nln_nnzj + 1):nln_nnzj] = ncl.nx + (1:ncl.meta.nnln)
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
  orig_lin_nnzj = ncl.nlp.meta.lin_nnzj
  orig_jvals = view(jvals, 1:orig_lin_nnzj)
  x = view(xr, 1:(ncl.nx))
  jac_lin_coord!(ncl.nlp, x, orig_jvals)
  if ncl.resid_linear
    jvals[(orig_lin_nnzj + 1):ncl.meta.lin_nnzj] .= 1
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
  orig_nln_nnzj = ncl.nlp.meta.nln_nnzj
  orig_jvals = view(jvals, 1:orig_nln_nnzj)
  x = view(xr, 1:(ncl.nx))
  jac_nln_coord!(ncl.nlp, x, orig_jvals)
  jvals[(orig_nln_nnzj + 1):ncl.meta.nln_nnzj] .= 1
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
    vr_lin = view(vr, ncl.meta.lin)
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
  vr_nl = ncl.resid_linear ? view(vr, ncl.meta.nln) : vr
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
    Jtv[ncl.nx .+ ncl.meta.lin] .= v
    Jtv[ncl.nx .+ ncl.meta.nln] .= 0
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
    Jtv[ncl.nx .+ ncl.meta.lin] .= 0
    Jtv[ncl.nx .+ ncl.meta.nln] .= v
  else
    Jtv[(ncl.nx + 1):(ncl.nx + ncl.meta.nnln)] .= v
  end
  return Jtv
end
