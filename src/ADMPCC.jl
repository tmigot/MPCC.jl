##############################################################################
#
# Specialization of AbstractMPCCModel based on automatic differentiation
# Based on ADNLPModel:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/autodiff_model.jl
#
# TODO:
# - doc
#
##############################################################################

@doc raw"""ADMPCCModel is an AbstractMPCCModel using ForwardDiff to compute the
derivatives.
In this interface, the objective function ``f`` and an initial estimate are
required. If there are constraints, the function
``c:ℝⁿ → ℝᵐ``  and the vectors
``c_L`` and ``c_U`` also need to be passed. Bounds on the variables and an
inital estimate to the Lagrangian multipliers can also be provided.
````
ADMPCCModel(f, x0; lvar = [-∞,…,-∞], uvar = [∞,…,∞], y0 = zeros,
  c = NotImplemented, lcon = [-∞,…,-∞], ucon = [∞,…,∞], name = "Generic")
````
  - `f` - The objective function ``f``. Should be callable;
  - `x0 :: AbstractVector` - The initial point of the problem;
  - `lvar :: AbstractVector` - ``ℓ``, the lower bound of the variables;
  - `uvar :: AbstractVector` - ``u``, the upper bound of the variables;
  - `c` - The constraints function ``c``. Should be callable;
  - `y0 :: AbstractVector` - The initial value of the Lagrangian estimates;
  - `lcon :: AbstractVector` - ``c_L``, the lower bounds of the constraints function;
  - `ucon :: AbstractVector` - ``c_U``, the upper bounds of the constraints function;
  - `name :: String` - A name for the model.
The functions follow the same restrictions of ForwardDiff functions, summarised
here:
  - The function can only be composed of generic Julia functions;
  - The function must accept only one argument;
  - The function's argument must accept a subtype of AbstractVector;
  - The function should be type-stable.
For contrained problems, the function ``c`` is required, and it must return
an array even when m = 1,
and ``c_L`` and ``c_U`` should be passed, otherwise the problem is ill-formed.
For equality constraints, the corresponding index of ``c_L`` and ``c_U`` should be the
same.
"""
mutable struct ADMPCCModel <: AbstractMPCCModel
  meta :: MPCCModelMeta

  counters :: MPCCCounters

  # Functions
  f
  c
  G
  H
end

function ADMPCCModel(f, x0::AbstractVector; y0::AbstractVector = eltype(x0)[],
                    lvar::AbstractVector = eltype(x0)[], uvar::AbstractVector = eltype(x0)[],
                    lcon::AbstractVector = eltype(x0)[], ucon::AbstractVector = eltype(x0)[],
                    c = (args...)->throw(NotImplementedError("cons")),
                    G = (args...)->throw(NotImplementedError("consG")),
                    H = (args...)->throw(NotImplementedError("consH")),
                    lccG::AbstractVector = eltype(x0)[], lccH::AbstractVector = eltype(x0)[],
                    name::String = "Generic", lin::AbstractVector{<: Integer}=Int[])

  nvar = length(x0)
  length(lvar) == 0 && (lvar = -Inf*ones(nvar))
  length(uvar) == 0 && (uvar =  Inf*ones(nvar))
  ncon = maximum([length(lcon); length(ucon); length(y0)])
  ncc  = length(lccG) #should be equal as length(lccH)

  A = ForwardDiff.hessian(f, x0)
  for i = 1:ncon
    A += ForwardDiff.hessian(x->c(x)[i], x0) * (-1)^i
  end
  nnzh = nvar * (nvar + 1) / 2
  nnzj = 0

  if ncon > 0
    length(lcon) == 0 && (lcon = -Inf*ones(ncon))
    length(ucon) == 0 && (ucon =  Inf*ones(ncon))
    length(y0) == 0   && (y0 = zeros(ncon))
    nnzj = nvar * ncon
  end
  nln = setdiff(1:ncon, lin)

  meta = MPCCModelMeta(nvar, x0 = x0, y0=y0, ncc = ncc, lccG = lccG, lccH = lccH,
                             lvar = lvar, uvar = uvar,
                             ncon = ncon, lcon = lcon, ucon = ucon,
                             nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln,
                             minimize=true, islp=false, name=name)

  return ADMPCCModel(meta, MPCCCounters(), f, c, G, H)
end

function obj(nlp :: ADMPCCModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.f(x)
end

function grad!(nlp :: ADMPCCModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  ForwardDiff.gradient!(view(g, 1:length(x)), nlp.f, x)
  return g
end

function cons_nl!(nlp :: ADMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncon] = nlp.c(x)
  return c
end

function consG!(nlp :: ADMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncc] = nlp.G(x)
  return c
end

function consH!(nlp :: ADMPCCModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  c[1:nlp.meta.ncc] = nlp.H(x)
  return c
end

function jac_nl_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : nlp.meta.nnzj] .= getindex.(I, 1)[:]
  cols[1 : nlp.meta.nnzj] .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_nl_coord!(nlp :: ADMPCCModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  Jx = ForwardDiff.jacobian(nlp.c, x)
  vals[1 : nlp.meta.nnzj] .= Jx[:]
  return vals
end

function jacG_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacG_coord!(nlp :: ADMPCCModel, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  increment!(nlp, :neval_jacG)
  Jx = ForwardDiff.jacobian(nlp.G, x)
  vals[1 : n*m] .= Jx[:]
  return vals
end

function jacH_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacH_coord!(nlp :: ADMPCCModel, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  increment!(nlp, :neval_jacH)
  Jx = ForwardDiff.jacobian(nlp.H, x)
  vals[1 : n*m] .= Jx[:]
  return vals
end

function jnlprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] = ForwardDiff.derivative(t -> nlp.c(x + t * v), 0)
  return Jv
end

function jnltprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
  return Jtv
end

function jGprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jGprod)
  Jv[1:nlp.meta.ncc] = ForwardDiff.derivative(t -> nlp.G(x + t * v), 0)
  return Jv
end

function jGtprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jGtprod)
  Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.G(x), v), x)
  return Jtv
end

function jHprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jGprod)
  Jv[1:nlp.meta.ncc] = ForwardDiff.derivative(t -> nlp.H(x + t * v), 0)
  return Jv
end

function jHtprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jGtprod)
  Jtv[1:nlp.meta.nvar] = ForwardDiff.gradient(x -> dot(nlp.H(x), v), x)
  return Jtv
end

#function hess(nlp :: ADMPCCModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

#function hess(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
#  increment!(nlp, :neval_hess)
#  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y)
#  Hx = ForwardDiff.hessian(ℓ, x)
#  return tril(Hx)
#end

function hess_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: ADMPCCModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * nlp.f(x)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessG(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessG)
  ℓ(x) = dot(nlp.G(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hessG_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessG_coord!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_hessG)
  ℓ(x) = dot(nlp.G(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessH(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessH)
  ℓ(x) = dot(nlp.H(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hessH_structure!(nlp :: ADMPCCModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessH_coord!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_hess)
  ℓ(x) = dot(nlp.H(x), y)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: ADMPCCModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ℓ(x) = obj_weight * nlp.f(x)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hprod!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  ℓ(x) = obj_weight * nlp.f(x) + dot(nlp.c(x), y[1:ncon]) - dot(nlp.G(x), y[ncon+1:ncon+ncc]) - dot(nlp.H(x), y[ncon+ncc+1:ncon+2*ncc])
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hGprod!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  increment!(nlp, :neval_hprod)
  ℓ(x) = dot(nlp.G(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end

function hHprod!(nlp :: ADMPCCModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  increment!(nlp, :neval_hprod)
  ℓ(x) = dot(nlp.H(x), y)
  Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
  return Hv
end
