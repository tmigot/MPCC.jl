mutable struct ADMPCCModel{T,S,Si,FG<:Function,FH<:Function} <: AbstractMPCCModel{T,S}
  nlp::ADNLPModels.ADNLPModel{T,S,Si}
  meta::NLPModelMeta{T,S}
  cc_meta::MPCCModelMeta{T,S}
  cc_counters::MPCCCounters

  # Functions
  G::FG
  H::FH
end

function ADMPCCModel(
  G::Function,
  H::Function,
  lccG::S,
  lccH::S,
  args...;
  yG::S = fill!(similar(lccG), zero(eltype(S))),
  yH::S = fill!(similar(lccG), zero(eltype(S))),
  kwargs...,
) where {S}
  nlp = ADNLPModels.ADNLPModel(args...; hessian_backend = ADNLPModels.ForwardDiffADHessian, kwargs...)
  nvar = nlp.meta.nvar
  meta = NLPModelMeta(
    nvar,
    x0 = nlp.meta.x0,
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    ncon = nlp.meta.ncon,
    y0 = nlp.meta.y0,
    lcon = nlp.meta.lcon,
    ucon = nlp.meta.ucon,
    nnzj = nlp.meta.nnzj,
    nln_nnzj = nlp.meta.nnzj,
    # nnzh = nnzh,
    # minimize = minimize,
    # islp = false,
    name = nlp.meta.name,
  )
  ncc = length(lccG)
  cc_meta = MPCCModelMeta(nvar, ncc, lccG = lccG, lccH = lccH, yG = yG, yH = yH)
  return ADMPCCModel(nlp, meta, cc_meta, MPCCCounters(), G, H)
end

for meth in (:obj, :grad!, :objgrad!, :objcons!, :jac_op!, :ghjvprod!, :jth_hprod!)
  @eval begin
    $meth(nlp::ADMPCCModel, args...; kwargs...) = $meth(nlp.nlp, args...; kwargs...)
  end
end
cons!(nlp::ADMPCCModel, x::AbstractVector, cx::AbstractVector) = cons!(nlp.nlp, x, cx)
jac_coord!(nlp::ADMPCCModel, x::AbstractVector, vals::AbstractVector) =
  jac_coord!(nlp.nlp, x, vals)
jac_structure!(nlp::ADMPCCModel, rows::AbstractVector, cols::AbstractVector) =
  jac_structure!(nlp.nlp, rows, cols)
jprod!(nlp::ADMPCCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector) =
  jprod!(nlp.nlp, x, v, Jv)
jtprod!(nlp::ADMPCCModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) =
  jtprod!(nlp.nlp, x, v, Jtv)

@default_cc_counters ADMPCCModel nlp

function NLPModels.reset!(nlp::ADMPCCModel)
  reset!(nlp.cc_counters)
  reset!(nlp.nlp)
end

function consG!(nlp::ADMPCCModel, x::AbstractVector, c::AbstractVector)
  increment_cc!(nlp, :neval_consG)
  c[1:nlp.cc_meta.ncc] .= nlp.G(x)
  return c
end

function consH!(nlp::ADMPCCModel, x::AbstractVector, c::AbstractVector)
  increment_cc!(nlp, :neval_consH)
  c[1:nlp.cc_meta.ncc] .= nlp.H(x)
  return c
end

function jacG_structure!(
  nlp::ADMPCCModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.cc_meta.ncc, nlp.meta.nvar
  I = ((i, j) for i = 1:m, j = 1:n)
  rows[1:n*m] .= getindex.(I, 1)[:]
  cols[1:n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacG_coord!(nlp::ADMPCCModel, x::AbstractVector, vals::AbstractVector)
  m, n = nlp.cc_meta.ncc, nlp.meta.nvar
  increment_cc!(nlp, :neval_jacG)
  Jx = ForwardDiff.jacobian(nlp.G, x)
  vals[1:n*m] .= Jx[:]
  return vals
end

function jacH_structure!(
  nlp::ADMPCCModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  m, n = nlp.cc_meta.ncc, nlp.meta.nvar
  I = ((i, j) for i = 1:m, j = 1:n)
  rows[1:n*m] .= getindex.(I, 1)[:]
  cols[1:n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacH_coord!(nlp::ADMPCCModel, x::AbstractVector, vals::AbstractVector)
  m, n = nlp.cc_meta.ncc, nlp.meta.nvar
  increment_cc!(nlp, :neval_jacH)
  Jx = ForwardDiff.jacobian(nlp.H, x)
  vals[1:n*m] .= Jx[:]
  return vals
end

function jGprod!(nlp::ADMPCCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  increment_cc!(nlp, :neval_jGprod)
  Jv[1:nlp.cc_meta.ncc] .= ForwardDiff.derivative(t -> nlp.G(x + t * v), 0)
  return Jv
end

function jGtprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment_cc!(nlp, :neval_jGtprod)
  Jtv[1:nlp.meta.nvar] .= ForwardDiff.gradient(x -> dot(nlp.G(x), v), x)
  return Jtv
end

function jHprod!(nlp::ADMPCCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  increment_cc!(nlp, :neval_jHprod)
  Jv[1:nlp.cc_meta.ncc] .= ForwardDiff.derivative(t -> nlp.H(x + t * v), 0)
  return Jv
end

function jHtprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment_cc!(nlp, :neval_jHtprod)
  Jtv[1:nlp.meta.nvar] .= ForwardDiff.gradient(x -> dot(nlp.H(x), v), x)
  return Jtv
end

function hess_structure!(
  nlp::ADMPCCModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1:nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1:nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nlp.meta.nvar
    for i = j:nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  increment!(nlp, :neval_hess)
  ncon, ncc = nlp.meta.ncon, nlp.cc_meta.ncc
  ℓ(x) = ADNLPModels.get_lag(nlp.nlp, nlp.nlp.adbackend.hessian_backend, obj_weight)(x) -
    dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nlp.meta.nvar
    for i = j:nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessG_structure!(
  nlp::ADMPCCModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1:nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1:nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessG_coord!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector,
)
  increment_cc!(nlp, :neval_hessG)
  ℓ(x) = dot(nlp.G(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nlp.meta.nvar
    for i = j:nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessH_structure!(
  nlp::ADMPCCModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = nlp.meta.nvar
  I = ((i, j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1:nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1:nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessH_coord!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector,
)
  increment!(nlp, :neval_hess)
  ℓ(x) = dot(nlp.H(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1:nlp.meta.nvar
    for i = j:nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  increment!(nlp, :neval_hprod)
  ncon, ncc = nlp.meta.ncon, nlp.cc_meta.ncc
  function c(x; ncon = ncon)
    cx = similar(x, ncon)
    nlp.nlp.c!(cx, x)
    return cx
  end
  ℓ(x) = ADNLPModels.get_lag(nlp.nlp, nlp.nlp.adbackend.hessian_backend, obj_weight)(x) -
    dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hGprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  increment_cc!(nlp, :neval_hGprod)
  ℓ(x) = dot(nlp.G(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hHprod!(
  nlp::ADMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  increment_cc!(nlp, :neval_hHprod)
  ℓ(x) = dot(nlp.H(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
