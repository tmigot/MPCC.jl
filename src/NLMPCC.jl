"""
Convert an MPCCModel to an NLPModels as follows.
Definit le type NLMPCC :
min 	f(x)
s.t. 	l <= x <= u
	lcon(tb) <= cnl(x) <= ucon

with

cnl(x) := c(x),G(x),H(x),G(x)H(x)
"""
mutable struct NLMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters

 x0       :: Vector

 mod      :: AbstractMPCCModel

 function NLMPCC(mod     :: AbstractMPCCModel;
                  x      :: Vector = mod.meta.x0)

  ncc = mod.meta.ncc

  n = mod.meta.nvar

  new_lcon = vcat(mod.meta.lcon,
                  mod.meta.lccG,
                  mod.meta.lccH,
                  -Inf*ones(ncc))
  new_ucon = vcat(mod.meta.ucon,
                  Inf*ones(2*ncc),
                  zeros(ncc))

  meta = NLPModelMeta(n, x0 = x, lvar = mod.meta.lvar, uvar = mod.meta.uvar,
                                 ncon = mod.meta.ncon+3*ncc,
                                 lcon = new_lcon, ucon = new_ucon)

  return new(meta, Counters(), x, mod)
 end
end

############################################################################

# Getteur

############################################################################

function obj(nlp :: NLMPCC, x :: Vector)
 return obj(nlp.mod, x)
end

function grad!(nlp :: NLMPCC, x :: Vector, gx :: Vector)
 return grad!(nlp.mod, x, gx)
end

#########################################################
#
# Return the vector of the constraints
# lb <= x <= ub,
# lc <= c(x) <= uc,
# lccG <= G(x), lccH <= H(x),
# G(x).*H(x) <= 0
#
#########################################################
function cons!(nlp :: NLMPCC, x :: AbstractVector, c :: AbstractVector)
  c = cons(nlp.mod, x)
  return c
end

#########################################################
#
# Return the jacobian matrix (ncon+2*ncc x n)
# [Jc(x), -JG(x), -JH(x)]
#
#########################################################

function jac(nlp :: NLMPCC, x :: Vector)

 A = vcat(jac_nl(nlp.mod,x),jacG(nlp.mod,x),jacH(nlp.mod,x),diag(consH(nlp.mod,x))*jacG(nlp.mod,x)+diag(consG(nlp.mod,x))*jacH(nlp.mod,x))

 return A
end

function jac_coord(nlp :: NLMPCC, x :: Vector)
 return findnz(jac(nlp.mod, x))
end

function jprod(nlp :: NLMPCC, x :: Vector, v :: Vector)
  return jac(nlp,x)*v
end

function jprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  Jv = jprod(mod, x, v)
  return Jv
end

function jtprod(nlp :: NLMPCC, x :: Vector, v :: Vector)
 return jac(nlp,x)'*v
end

function jtprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  Jtv = jtprod(mod, x, v)
  return Jtv
end

function hess(nlp :: NLMPCC, x :: Vector ; obj_weight = 1.0, y = zeros)

 if y != zeros && nlp.mod.meta.ncc >0
  ncc, nlc = nlp.mod.meta.ncc, nlp.mod.meta.ncon
  Hx=hessnl(nlp.mod, x, obj_weight = obj_weight, y = y[1:nlc])
  HG=hessG(nlp.mod,x,obj_weight=0.0,y=y[nlc+1:ncc])
  HH=hessH(nlp.mod,x,obj_weight=0.0,y=y[nlc+ncc+1:nlc+2*ncc])
  JGJH=jacG(nlp.mod,x)'*jacH(nlp.mod,x)
  GGH=hessG(nlp.mod,x,obj_weight=0.0,y=consH(nlp.mod,x).*y[nlc+2*ncc+1:nlc+3*ncc])
  HHG=hessG(nlp.mod,x,obj_weight=0.0,y=consH(nlp.mod,x).*y[nlc+2*ncc+1:nlc+3*ncc])
  rslt = Hx+HG+HH+2*JGJH+GGH+HHG
 elseif y != zeros
  Hx=hessnl(nlp.mod, x, obj_weight = obj_weight, y = y)
  #HG=hessG(nlp.mod,x,obj_weight=0.0)
  #HH=hessH(nlp.mod,x,obj_weight=0.0)
  #JGJH=jacG(nlp.mod,x)'*jacH(nlp.mod,x)
  #GGH=hessG(nlp.mod,x,obj_weight=0.0,y=consH(nlp.mod,x))
  #HHG=hessG(nlp.mod,x,obj_weight=0.0,y=consH(nlp.mod,x))
  rslt = Hx
else #y == zeros
  rslt = hess(nlp.mod, x)
 end

 return rslt
end

function hess_coord(nlp :: NLMPCC, x :: AbstractVector; obj_weight = 1.0,
      y :: AbstractVector = zeros(nlp.meta.ncon))
  Hx = hess(nlp, x, obj_weight=obj_weight, y=y)
  if isa(Hx, SparseMatrixCSC)
    return findnz(Hx) #findnz(Hx, SparseMatrixCSC)
  else
    I = findall(!iszero, Hx)
    return (getindex.(I, 1), getindex.(I, 2), Hx[I])
    #return findnz(sparse(Hx))
  end
end

function hprod(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = zeros(nlp.meta.ncon))
 return hess(nlp, x, obj_weight=obj_weight,y=y)*v
end

function hprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector;
    obj_weight = 1.0, y :: AbstractVector = zeros(nlp.meta.ncon))
  Hv = hess(nlp, x, obj_weight=obj_weight,y=y)*v
  return Hv
end

"""
Return the violation of the constraints
lb <= x <= ub,
lc <= c(x) <= uc,
lccG <= G(x),
lccH <= H(x),
G(x).*H(x) <= 0.
"""
function viol(nlp :: NLMPCC, x :: AbstractVector)

 mod = nlp.mod
 n  = mod.meta.nvar
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.meta.lvar-x, 0), max.(x-mod.meta.uvar, 0))

 if mod.meta.ncon !=0

  c = cons_nl(mod, x)
  feas_c = vcat(max.(mod.meta.lcon-c, 0), max.(c-mod.meta.ucon, 0))

 else

  feas_c = Float64[]

 end

 if ncc != 0

  G=consG(mod, x)
  H=consH(mod, x)

  feas_cp = vcat(max.(mod.meta.lccG-G, 0), max.(mod.meta.lccH-H, 0))
  feas_cc = max.(G.*H, 0)
 else
  feas_cp = Float64[]
  feas_cc = Float64[]
 end

 return vcat(feas_x, feas_c, feas_cp, feas_cc)
end
