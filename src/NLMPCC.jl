"""
Convert an MPCCModel to an NLPModels as follows.
Definit le type NLMPCC :
min 	f(x)
s.t. 	l <= x <= u
	lcon(tb) <= cnl(x) <= ucon

with

cnl(x) := c(x),G(x),H(x),G(x).*H(x)
"""
mutable struct NLMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters

 x0       :: Vector

 mod      :: AbstractMPCCModel

 function NLMPCC(mod     :: AbstractMPCCModel;
                  x      :: Vector = mod.meta.x0)

  n, ncc = mod.meta.nvar, mod.meta.ncc
  ncon = maximum([length(mod.meta.lcon); length(mod.meta.ucon); length(mod.meta.y0)])
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

function obj(nlp :: NLMPCC, x ::  AbstractVector)
 increment!(nlp, :neval_obj)
 return obj(nlp.mod, x)
end

function grad!(nlp :: NLMPCC, x :: Vector, gx :: AbstractVector)
 increment!(nlp, :neval_grad)
 return grad!(nlp.mod, x, gx)
end

function cons!(nlp :: NLMPCC, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  Gx, Hx = consG(nlp.mod, x), consH(nlp.mod, x)
  c[1:nlp.meta.ncon] = vcat(cons_nl(nlp.mod, x), Gx, Hx, Gx .* Hx)
  return c
end

function jac(nlp :: NLMPCC, x :: AbstractVector)
 increment!(nlp, :neval_jac)
 JGx, JHx = jacG(nlp.mod,x),   jacH(nlp.mod,x)
 Gx,  Hx  = consG(nlp.mod, x), consH(nlp.mod, x)
 A = vcat(jac_nl(nlp.mod,x), JGx, JHx, diagm(0 => Hx) * JGx + diagm(0 => Gx) * JHx)

 return A
end

function jac_coord!(nlp :: NLMPCC, x :: AbstractVector, vals ::AbstractVector)
 Jx = jac(nlp, x) #findnz(jac(nlp.mod, x))
 m, n = size(Jx)
 vals[1 : n*m] .= Jx[:]
 return vals
end

function jac_structure!(nlp :: NLMPCC, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
 m, n = nlp.meta.ncon, nlp.meta.nvar
 I = ((i,j) for i = 1:m, j = 1:n)
 rows[1 : n*m] .= getindex.(I, 1)[:]
 cols[1 : n*m] .= getindex.(I, 2)[:]
 return rows, cols
end

function jprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv = jac(nlp, x) * v
  return Jv
end

function jtprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv = jac(nlp,x)' * v
  return Jtv
end

function hess(nlp :: NLMPCC, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp.mod, x, obj_weight = obj_weight)
  return tril(Hx)
end

function hess(nlp :: NLMPCC, x :: AbstractVector, y :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncc, nlc, nvar = nlp.mod.meta.ncc, nlp.mod.meta.ncon, nlp.mod.meta.nvar
  JGx, JHx = jacG(nlp.mod,x),   jacH(nlp.mod,x)
  Gx,  Hx  = consG(nlp.mod, x), consH(nlp.mod, x)
  lphi = y[nlc+2*ncc+1:nlc+3*ncc]
  ny = vcat(y[1:nlc], y[nlc+1:nlc+ncc] - lphi .* Hx, y[nlc+ncc+1:nlc+2*ncc] - lphi .* Gx)
  Hx = hess(nlp.mod, x, ny, obj_weight = obj_weight)
  for i=1:ncc
   Hx += lphi[i] * (JGx[i,:] * JHx[i,:]' + JHx[i,:] * JGx[i,:]')
  end
  return tril(Hx)
end

function hess_structure!(nlp :: NLMPCC, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: NLMPCC, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
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

function hess_coord!(nlp :: NLMPCC, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
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

function hprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, obj_weight=obj_weight)
  Hv = (Hx + Hx' - diagm(0 => diag(Hx))) * v
  return Hv
end

function hprod!(nlp :: NLMPCC, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, y, obj_weight=obj_weight)
  Hv = (Hx + Hx' - diagm(0 => diag(Hx))) * v
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

 mod, n, ncc = nlp.mod, nlp.mod.meta.nvar, nlp.mod.meta.ncc
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
