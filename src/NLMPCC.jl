"""
Convert an MPCCModel to an NLPModels as follows.
Definit le type NLMPCC :
min 	f(x)
s.t. 	l <= x <= u
	lcon(tb) <= cnl(x) <= ucon

with

cnl(x) := c(x),G(x),H(x),G(x).*H(x)
"""
mutable struct NLMPCC{T, S} <: AbstractNLPModel{T, S}

 meta     :: NLPModelMeta{T, S}
 counters :: Counters
 mod      :: AbstractMPCCModel{T, S}

end

function NLMPCC(mod:: AbstractMPCCModel{T, S}; kwargs...) where {T, S}

  nvar, ncc = mod.meta.nvar, mod.cc_meta.ncc
  new_lcon = vcat(mod.meta.lcon,
    mod.cc_meta.lccG,
    mod.cc_meta.lccH,
    -Inf*ones(ncc))
  new_ucon = vcat(mod.meta.ucon,
    Inf*ones(2*ncc),
    zeros(ncc))
  ncon = maximum([length(new_lcon); length(new_ucon)])
  y0 = vcat(mod.meta.y0, zeros(3*ncc))

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon

  meta = NLPModelMeta(nvar, x0 = mod.meta.x0, lvar = mod.meta.lvar, uvar = mod.meta.uvar,
                      ncon = mod.meta.ncon+3*ncc, y0=y0,
                      lcon = new_lcon, ucon = new_ucon,
      nnzj=nnzj, nnzh=nnzh, lin=mod.meta.lin,
      minimize=mod.meta.minimize, islp=false, name="Nonlinear variant of $(mod.meta.name)")

  return NLMPCC(meta, Counters(), mod)
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
  c[1:nlp.meta.ncon] .= vcat(cons(nlp.mod, x), Gx, Hx, Gx .* Hx)
  return c
end

function jac_coord!(nlp :: NLMPCC, x :: AbstractVector, vals ::AbstractVector)
 JGx, JHx = jacG(nlp.mod,x),   jacH(nlp.mod,x)
 Gx,  Hx  = consG(nlp.mod, x), consH(nlp.mod, x)
 A = vcat(jac(nlp.mod,x), JGx, JHx, diagm(0 => Hx) * JGx + diagm(0 => Gx) * JHx)
 m, n = size(A)
 vals[1 : n*m] .= A[:]
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
  Jv .= jac(nlp, x) * v
  return Jv
end

function jtprod!(nlp :: NLMPCC, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= jac(nlp, x)' * v
  return Jtv
end

function hess_structure!(nlp :: NLMPCC, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: NLMPCC, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  hess_coord!(nlp.mod, x, vals, obj_weight=obj_weight)
end

function hess_coord!(nlp :: NLMPCC, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ##############################################
  ncc, nlc, nvar = nlp.mod.cc_meta.ncc, nlp.mod.meta.ncon, nlp.mod.meta.nvar
  JGx, JHx = jacG(nlp.mod,x),   jacH(nlp.mod,x)
  Gx,  Hx  = consG(nlp.mod, x), consH(nlp.mod, x)
  lphi = y[nlc+2*ncc+1:nlc+3*ncc]
  ny = vcat(y[1:nlc], y[nlc+1:nlc+ncc] - lphi .* Hx, y[nlc+ncc+1:nlc+2*ncc] - lphi .* Gx)
  Hx = hess(nlp.mod, x, ny, obj_weight = obj_weight)
  for i=1:ncc
   Hx += lphi[i] * (JGx[i,:] * JHx[i,:]' + JHx[i,:] * JGx[i,:]')
  end
  Hx = tril(Hx)
  #############################################
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
  hprod!(nlp.mod, x, v, Hv, obj_weight=obj_weight)
end

function hprod!(nlp :: NLMPCC, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hx = hess(nlp, x, y, obj_weight=obj_weight)
  Hv .= (Hx + Hx' - diagm(0 => diag(Hx))) * v
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

 mod, n, ncc = nlp.mod, nlp.mod.meta.nvar, nlp.mod.cc_meta.ncc
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.meta.lvar-x, 0), max.(x-mod.meta.uvar, 0))

 if mod.meta.ncon !=0

  c = cons(mod, x)
  feas_c = vcat(max.(mod.meta.lcon-c, 0), max.(c-mod.meta.ucon, 0))

 else

  feas_c = Float64[]

 end

 if ncc != 0

  G=consG(mod, x)
  H=consH(mod, x)

  feas_cp = vcat(max.(mod.cc_meta.lccG-G, 0), max.(mod.cc_meta.lccH-H, 0))
  feas_cc = max.(G.*H, 0)
 else
  feas_cp = Float64[]
  feas_cc = Float64[]
 end
 return vcat(feas_x, feas_c, feas_cp, feas_cc)
end
