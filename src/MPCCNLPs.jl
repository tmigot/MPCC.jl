##############################################################################
#
# Specialization of AbstractMPCCModel when mp, G, H are given by NLPModels
#
# TODO:
# - weird cons function
# - jac, jacG, jacH
# - check hess operators
#
##############################################################################

"""
Definit le type MPCC :
min f(x)
l <= x <= u
lb <= c(x) <= ub
lccG <= G(x) _|_ H(x) >= lccH
"""
mutable struct MPCCNLPs <: AbstractMPCCModel

 mp :: AbstractNLPModel
 G  :: AbstractNLPModel
 H  :: AbstractNLPModel

 meta :: MPCCModelMeta

 counters :: MPCCCounters

 function MPCCNLPs(mp  :: AbstractNLPModel;
                   G   :: AbstractNLPModel = ADNLPModel(x->0, mp.meta.x0),
                   H   :: AbstractNLPModel = ADNLPModel(x->0, mp.meta.x0),
                   ncc :: Int64 = -1,
                   x0  :: Vector = mp.meta.x0)

  ncc = ncc == -1 ? G.meta.ncon : ncc

  if G.meta.ncon != H.meta.ncon || (ncc != -1 && G.meta.ncon != ncc)
   throw(error("Incompatible complementarity"))
  end

  n    = length(x0)
  ncon = mp.meta.ncon

  meta = MPCCModelMeta(n, x0 = x0, ncc = ncc, lccG = G.meta.lcon, lccH = H.meta.lcon,
                              lvar = mp.meta.lvar, uvar = mp.meta.uvar,
                              ncon = ncon,
                              lcon = mp.meta.lcon, ucon = mp.meta.ucon)

  return new(mp, G, H, meta, MPCCCounters())
 end
end

############################################################################

# Getteur

############################################################################
"""
Evaluate f(x), the objective function at x.
"""
function obj(mod :: MPCCNLPs, x :: AbstractVector)
 increment!(mod, :neval_obj)
 return obj(mod.mp, x)
end

"""
Evaluate ∇f(x), the gradient of the objective function at x in place.
"""
function grad!(mod :: MPCCNLPs, x :: Vector, gx :: AbstractVector)
 increment!(mod, :neval_grad)
 return grad!(mod.mp, x, gx)#gx
end

"""
Evaluate ``c(x)``, the constraints at `x`.
"""
function cons_nl!(mod :: MPCCNLPs, x :: Vector, c :: AbstractVector)
 increment!(mod, :neval_cons)
 return cons!(mod.mp, x, c)
end

"""
Evaluate ``G(x)``, the constraints at `x`.
"""
function consG!(mod :: MPCCNLPs, x :: Vector, c :: AbstractVector)
 increment!(mod, :neval_consG)
 if mod.meta.ncc > 0
  c .= cons(mod.G, x)
 else
  c .= Float64[]
 end

 return c
end

"""
Evaluate ``H(x)``, the constraints at `x`.
"""
function consH!(mod :: MPCCNLPs, x :: Vector, c :: AbstractVector)
 increment!(mod, :neval_consH)
 if mod.meta.ncc > 0
  c .= cons(mod.H, x)
 else
  c .= Float64[]
 end

 return c
end

"""
    Jx = jac(nlp, x)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]``, the constraint's Jacobian at `x` as a sparse matrix.
Size of the jacobian matrix: (ncon + 2 * ncc) x n
"""
function jac(mod :: MPCCNLPs, x :: AbstractVector)
  n, ncon, ncc = mod.meta.nvar, mod.meta.ncon, mod.meta.ncc

 if ncon+ncc == 0

  A = sparse(zeros(0,n))

 else

  if ncon > 0
   J = jac_nl(mod, x)
  else
   J = sparse(zeros(0,n))
  end

  if ncc > 0
   JG, JH = jacG(mod,x), jacH(mod,x)
   A = [J;-JG;-JH]
  else
   A = J
  end

 end

 return A
end

"""
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jac_nl(mod :: MPCCNLPs, x :: AbstractVector)
 increment!(mod, :neval_jac)
 return jac(mod.mp, x)
end

"""
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacG(mod :: MPCCNLPs, x :: AbstractVector)
 increment!(mod, :neval_jacG)
 if mod.meta.ncc > 0
  rslt = jac(mod.G, x)
 else
  rslt = Float64[]
 end

 return rslt
end

"""
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacH(mod :: MPCCNLPs, x :: AbstractVector)
 increment!(mod, :neval_jacH)
 if mod.meta.ncc > 0
  rslt = jac(mod.H, x)
 else
  rslt = Float64[]
 end

 return rslt
end

"""
Jacobienne des contraintes actives à precmpcc près

(Tangi: copie de colonne si contrainte d'égalité?)
"""
function jacl(mod :: MPCCNLPs, x :: Vector) #bad name
 A, Il,Iu,Ig,Ih,IG,IH = jac_actif(mod, x, 0.0)
 return A
end

function jac_actif(mod :: MPCCNLPs, x :: Vector, prec :: Float64)

  n = mod.meta.nvar
  ncc = mod.meta.ncc

  Il = findall(z->z<=prec,abs.(x-mod.meta.lvar))
  Iu = findall(z->z<=prec,abs.(x-mod.meta.uvar))
  jl = zeros(n);jl[Il] .= 1.0;Jl=diagm(0 => jl);
  ju = zeros(n);ju[Iu] .= 1.0;Ju=diagm(0 => ju);
  IG=Int64[];IH=Int64[];Ig=Int64[];Ih=Int64[];

 if mod.meta.ncon+ncc ==0

  A=[]

 else

  if mod.meta.ncon > 0
   c = cons_nl(mod,x)
   J = jac_nl(mod, x)
  else
   c = Float64[]
   J = sparse(zeros(0,2))
  end

  Ig=findall(z->z<=prec,abs.(c-mod.meta.lcon))
  Ih=findall(z->z<=prec,abs.(c-mod.meta.ucon))

  Jg, Jh = zeros(mod.meta.ncon,n), zeros(mod.meta.ncon,n)

  Jg[Ig,1:n] = J[Ig,1:n]
  Jh[Ih,1:n] = J[Ih,1:n]

  if ncc>0

   IG=findall(z->z<=prec,abs.(consG(mod,x)-mod.meta.lccG))
   IH=findall(z->z<=prec,abs.(consH(mod,x)-mod.meta.lccH))

   JG, JH = zeros(ncc, n), zeros(ncc, n)
   JG[IG,1:n] = jacG(mod,x)[IG,1:n]
   JH[IH,1:n] = jacH(mod,x)[IH,1:n]

   A=[Jl;Ju;-Jg;Jh;-JG;-JH]'
  else
   A=[Jl;Ju;-Jg;Jh]'
  end

 end

 return A, Il,Iu,Ig,Ih,IG,IH
end

"""
    Jv = jnlprod(nlp, x, v, Jtv)
Evaluate ``∇c(x)v``, the transposed-Jacobian-vector product at `x`.
"""
function jnlprod!(mod :: MPCCNLPs, x :: Vector, v :: Vector, Jv :: Vector)
 return jprod!(mod.mp, x, v, Jv)
end

"""
  JGv = jGprod!(nlp, x, v, Jv)
Evaluate ``∇G(x)v``, the Jacobian-vector product at `x` in place.
"""
function jGprod!(mod :: MPCCNLPs, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
    increment!(mod, :neval_jGprod)
    if mod.meta.ncc > 0
     Jv .= jprod(mod.G, x, v)
    else
     Jv .= Float64[]
    end
    return Jv
end

"""
  JHv = jHprod!(nlp, x, v, Jv)
Evaluate ``∇H(x)v``, the Jacobian-vector product at `x` in place.
"""
function jHprod!(mod :: MPCCNLPs, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
    increment!(mod, :neval_jHprod)
    if mod.meta.ncc > 0
     Jv .= jprod(mod.H, x, v)
    else
     Jv .= Float64[]
    end
    return Jv
end

"""
    Jtv = jtprod(nlp, x, v, Jtv)
Evaluate ``∇c(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jnltprod!(mod :: MPCCNLPs, x :: Vector, v :: Vector, Jv :: Vector)
 increment!(mod, :neval_jtprod)
 return jtprod!(mod.mp, x, v, Jv)
end

"""
    Jtv = jtprodG(nlp, x, v, Jtv)
Evaluate ``∇G(x)^Tv``, the transposed-Jacobian-vector product at `x`
"""
function jGtprod!(mod :: MPCCNLPs, x :: Vector, v :: Vector, Jv :: Vector)
 increment!(mod, :neval_jGtprod)
 if mod.meta.ncc > 0
  Jv .= jtprod(mod.G, x, v)
 else
  Jv .= Float64[]
 end

 return Jv
end

"""
    Jtv = jtprodH(nlp, x, v, Jtv)
Evaluate ``∇H(x)^Tv``, the transposed-Jacobian-vector product at `x`
"""
function jHtprod!(mod :: MPCCNLPs, x :: Vector, v :: Vector, Jv :: Vector)
 increment!(mod, :neval_jHtprod)
 if mod.meta.ncc > 0
  Jv .= jtprod(mod.H, x, v)
 else
  Jv .= Float64[]
 end

 return Jv
end

function hess(mod :: MPCCNLPs, x :: Vector ; obj_weight = 1.0)
 increment!(mod, :neval_hess)
 return hess(mod.mp, x, obj_weight = obj_weight)
end

function hess(mod :: MPCCNLPs, x :: AbstractVector, y :: AbstractVector; obj_weight = 1.0)
 increment!(mod, :neval_hess)
 ncon, ncc = mod.meta.ncon, mod.meta.ncc
 yG, yH = y[ncon+1:ncon+ncc], y[ncon+ncc+1:ncon+2*ncc]
 return hess(mod.mp, x, y[1:ncon], obj_weight = obj_weight) - hessG(mod, x, yG) - hessH(mod, x, yH)
end

function hessG(mod :: MPCCNLPs, x :: AbstractVector, y :: AbstractVector; obj_weight = 1.0)
 increment!(mod, :neval_hessG)
 if mod.meta.ncc > 0
  rslt = hess(mod.G, x, y, obj_weight = obj_weight)
 else
  rslt = zeros(0,0)
 end
 return rslt
end

function hessH(mod :: MPCCNLPs, x :: AbstractVector, y :: AbstractVector; obj_weight = 1.0)
 increment!(mod, :neval_hessH)
 if mod.meta.ncc > 0
  rslt = hess(mod.H, x, y, obj_weight = obj_weight)
 else
  rslt = zeros(0,0)
 end
 return rslt
end

function hess_structure!(mod :: MPCCNLPs, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = mod.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: MPCCNLPs, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp, x, obj_weight=obj_weight)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: MPCCNLPs, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp, x, y, obj_weight=obj_weight)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: MPCCNLPs, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, obj_weight = obj_weight)
  Hv .= (Hx + Hx' - diagm(0 => diag(Hx))) * v
  return Hv
end

function hprod!(nlp :: MPCCNLPs, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, y, obj_weight = obj_weight)
  Hv .= (Hx + Hx' - diagm(0 => diag(Hx))) * v
  return Hv
end
