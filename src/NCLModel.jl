# TODO: accept maximization problems

import NLPModels: increment!

export NCLModel

"""
    NCLModel(nlp)

Subtype of `AbstractNLPModel` designed to represent an NCL subproblem.
A general problem of the form

    minimize   f(x)
    subject to lvar ≤ x ≤ uvar
               lcon ≤ c(x) ≤ ucon

is transformed into

    mininmize  f(x) + λ'r + ρ ‖r‖²
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
mutable struct NCLModel{T,S,M} <: AbstractNLPModel{T,S} where {M<:AbstractNLPModel{T,S}}
    nlp::M
    nx::Int  # number of variables in nlp
    nr::Int  # number of residuals in nlp problem (nr = length(nlp.meta.nln))
    resid_linear::Bool

    meta::NLPModelMeta{T,S}
    counters::Counters

    y::S
    ρ::T # penalty parameter
end

# constructor
function NCLModel(
    nlp::AbstractNLPModel{T,S};
    resid::T = zero(T),
    resid_linear::Bool = true,
    ρ::T = one(T),
    y::S = fill!(similar(nlp.meta.x0, resid_linear ? nlp.meta.ncon : nlp.meta.nnln), 1),
) where {T,S}

    if (nlp.meta.ncon == 0)
        @warn("input problem $(nlp.meta.name) is unconstrained, not generating NCL model")
        return nlp
    elseif ((nlp.meta.nnln == 0) & !resid_linear)
        @warn(
            "input problem $(nlp.meta.name) is linearly constrained and `resid_linear` is `false`, snot generating NCL model"
        )
        return nlp
    end

    # number of residuals
    nr = resid_linear ? nlp.meta.ncon : nlp.meta.nnln

    # construct meta
    nx = nlp.meta.nvar
    nvar = nx + nr
    meta = NLPModelMeta{T,S}(
        nvar;
        lvar = vcat(nlp.meta.lvar, fill!(similar(nlp.meta.x0, nr), -Inf)),  # no bounds on residuals
        uvar = vcat(nlp.meta.uvar, fill!(similar(nlp.meta.x0, nr), Inf)),
        x0 = vcat(nlp.meta.x0, fill!(similar(nlp.meta.x0, nr), resid)),
        y0 = nlp.meta.y0,
        name = "NCL-" * nlp.meta.name,
        nnzj = nlp.meta.nnzj + nr,
        nnzh = nlp.meta.nnzh + nr,
        ncon = nlp.meta.ncon,
        lcon = nlp.meta.lcon,
        ucon = nlp.meta.ucon,
        minimize = true,  # nlp.meta.minimize,
        # TODO: define nln, etc.
    )

    nlp.meta.minimize || error("only minimization problems are currently supported")
    return NCLModel{T,S,typeof(nlp)}(nlp, nx, nr, resid_linear, meta, Counters(), y, ρ)
end

function NLPModels.obj(ncl::NCLModel{T,S,M}, xr::S) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_obj)
    x = view(xr, 1:ncl.nx)
    r = view(xr, ncl.nx+1:ncl.nx+ncl.nr)
    obj_val = obj(ncl.nlp, x)
    ncl.nlp.meta.minimize || (obj_val *= -1)
    obj_res = ncl.y' * r + ncl.ρ * dot(r, r) / 2
    # ncl.meta.minimize || (obj_res *= -1)
    return obj_val + obj_res
end

function NLPModels.grad!(
    ncl::NCLModel{T,S,M},
    xr::S,
    gx::S,
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_grad)
    x = view(xr, 1:ncl.nx)
    grad!(ncl.nlp, x, gx)
    ncl.nlp.meta.minimize || (gx[1:ncl.nx] .*= -1)
    r = view(xr, ncl.nx+1:ncl.nx+ncl.nr)
    gx[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * r .+ ncl.y
    # ncl.meta.minimize || (gx[ncl.nx + 1 : ncl.nx + ncl.nr] .*= -1)
    return gx
end

function NLPModels.hess_structure!(
    ncl::NCLModel{T,S,M},
    hrows::AbstractVector{<:Integer},
    hcols::AbstractVector{<:Integer},
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_hess)
    hess_structure!(ncl.nlp, hrows, hcols)
    orig_nnzh = ncl.nlp.meta.nnzh
    nnzh = ncl.meta.nnzh
    hrows[orig_nnzh+1:nnzh] .= ncl.nx+1:ncl.meta.nvar
    hcols[orig_nnzh+1:nnzh] .= ncl.nx+1:ncl.meta.nvar
    return (hrows, hcols)
end

function NLPModels.hess_coord!(
    ncl::NCLModel{T,S,M},
    xr::S,
    vals::S;
    obj_weight::T = one(T),
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_hess)
    nnzh = ncl.meta.nnzh
    orig_nnzh = ncl.nlp.meta.nnzh
    x = view(xr, 1:ncl.nx)
    hess_coord!(ncl.nlp, x, hvals; obj_weight = obj_weight)
    ncl.nlp.meta.minimize || (hvals[1:orig_nnzh] .*= -1)
    hvals[orig_nnzh+1:nnzh] .= ncl.ρ
    # if ncl.meta.minimize
    #   hvals[orig_nnzh + 1 : nnzh] .= ncl.ρ
    # else
    #   hvals[orig_nnzh + 1 : nnzh] .= -ncl.ρ
    # end
    return hvals
end

function NLPModels.hess_coord!(
    ncl::NCLModel{T,S,M},
    xr::S,
    y::S,
    hvals::S;
    obj_weight::T = one(T),
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_hess)
    nnzh = ncl.meta.nnzh
    orig_nnzh = ncl.nlp.meta.nnzh
    x = view(xr, 1:ncl.nx)
    hess_coord!(ncl.nlp, x, y, hvals; obj_weight = obj_weight)
    ncl.nlp.meta.minimize || (hvals[1:orig_nnzh] .*= -1)
    hvals[orig_nnzh+1:nnzh] .= ncl.ρ
    # if ncl.meta.minimize
    #   hvals[orig_nnzh + 1 : nnzh] .= ncl.ρ
    # else
    #   hvals[orig_nnzh + 1 : nnzh] .= -ncl.ρ
    # end
    return hvals
end

function NLPModels.hprod!(
    ncl::NCLModel{T,S,M},
    xr::S,
    v::S,
    hv::S;
    obj_weight::T = one(T),
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_hprod)
    x = view(xr, 1:ncl.nx)
    hprod!(ncl.nlp, x, view(v, 1:ncl.nx), hv; obj_weight = obj_weight)
    ncl.nlp.meta.minimize || (hv[1:ncl.nx] .*= -1)
    hv[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr]
    # if ncl.meta.minimize
    #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
    # else
    #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= -ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
    # end
    return hv
end

function NLPModels.hprod!(
    ncl::NCLModel{T,S,M},
    xr::S,
    y::S,
    v::S,
    hv::S;
    obj_weight::T = one(T),
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_hprod)
    x = view(xr, 1:ncl.nx)
    hprod!(ncl.nlp, x, y, view(v, 1:ncl.nx), hv; obj_weight = obj_weight)
    ncl.nlp.meta.minimize || (hv[1:ncl.nx] .*= -1)
    hv[ncl.nx+1:ncl.nx+ncl.nr] .= ncl.ρ * v[ncl.nx+1:ncl.nx+ncl.nr]
    # if ncl.meta.minimize
    #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
    # else
    #   hv[ncl.nx + 1 : ncl.nx + ncl.nr] .= -ncl.ρ * v[ncl.nx + 1 : ncl.nx + ncl.nr]
    # end
    return hv
end

function NLPModels.cons!(
    ncl::NCLModel{T,S,M},
    xr::S,
    cx::S,
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_cons)
    x = view(xr, 1:ncl.nx)
    cons!(ncl.nlp, x, cx)
    r = view(xr, ncl.nx+1:ncl.nx+ncl.nr)
    if ncl.resid_linear
        cx .+= r
    else
        cx[ncl.nlp.meta.nln] .+= r
    end

    return cx
end

function NLPModels.jac_structure!(
    ncl::NCLModel{T,S,M},
    jrows::AbstractVector{<:Integer},
    jcols::AbstractVector{<:Integer},
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_jac)
    jac_structure!(ncl.nlp, jrows, jcols)
    orig_nnzj = ncl.nlp.meta.nnzj
    nnzj = ncl.meta.nnzj
    jrows[orig_nnzj+1:nnzj] .= ncl.resid_linear ? (1:ncl.meta.ncon) : ncl.nlp.meta.nln
    jcols[orig_nnzj+1:nnzj] .= ncl.nx+1:ncl.meta.nvar
    return jrows, jcols
end

function NLPModels.jac_coord!(
    ncl::NCLModel{T,S,M},
    xr::S,
    jvals::S,
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_jac)
    x = view(xr, 1:ncl.nx)
    jac_coord!(ncl.nlp, x, jvals)
    jvals[ncl.nlp.meta.nnzj+1:ncl.meta.nnzj] .= 1
    return jvals
end

function NLPModels.jprod!(
    ncl::NCLModel{T,S,M},
    xr::S,
    v::S,
    Jv::S,
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_jprod)
    x = view(xr, 1:ncl.nx)
    vx = view(v, 1:ncl.nx)
    jprod!(ncl.nlp, x, vx, Jv)

    vr = view(v, ncl.nx+1:ncl.nx+ncl.nr)
    resv = zeros(eltype(Jv), ncl.meta.ncon)
    if ncl.resid_linear
        Jv .+= vr
    else
        Jv[ncl.nlp.meta.nln] .+= vr
    end
    return Jv
end

function NLPModels.jtprod!(
    ncl::NCLModel{T,S,M},
    xr::S,
    v::S,
    Jtv::S,
) where {T,S,M<:AbstractNLPModel{T,S}}
    increment!(ncl, :neval_jtprod)
    x = view(xr, 1:ncl.nx)
    jtprod!(ncl.nlp, x, v, Jtv)
    Jtv[ncl.nx+1:ncl.meta.nvar] .= (ncl.resid_linear) ? v : v[ncl.nlp.meta.nln]
    return Jtv
end
