"""
Definit le type MPCC :
min f(x)
l <= x <= u
lb <= c(x) <= ub

with an array of integer "cvar" of the same size as c(x).
cvar[i] is > 0 if c_i(x) is involved in a complementarity constraint:
(lb_i = c_i(x) or c_i(x) = ub_i) OR x[cvar[i]] = 0.

Note: cons(nlp, x) call twice cons(nlp.mp, x) [TO FIX - idem jac, hess]
"""
mutable struct MPCCAmpl <: AbstractMPCCModel

 mp   :: AbstractNLPModel
 cvar :: AbstractVector{Int64}

 meta :: MPCCModelMeta

 counters :: MPCCCounters

 function MPCCAmpl(mp   :: AbstractNLPModel,
                   cvar :: AbstractVector{Int64},
                   ncc  :: Int64;
                   x0   :: Vector = mp.meta.x0,
                   name::String = "Generic",
                   lin::AbstractVector{<: Integer}=Int[])

  nvar    = length(x0)
  ncon = mp.meta.ncon

  Icc = findall(x -> x>0, cvar)
  I   = findall(x -> x==0, cvar)

  nnzh = nvar * (nvar + 1) / 2
  nnzj = nvar * ncon
  nln = setdiff(1:ncon, lin)

  meta = MPCCModelMeta(nvar, x0 = x0, ncc = ncc,  y0=mp.meta.y0,
                             lccG = mp.meta.lcon[Icc],
                             lccH = mp.meta.lvar[cvar[Icc]],
                             lvar = mp.meta.lvar, uvar = mp.meta.uvar,
                             ncon = ncon,
                             lcon = mp.meta.lcon[I], ucon = mp.meta.ucon[I],
                             nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln,
                             minimize=true, islp=false, name=name)

  return new(mp, cvar, meta, MPCCCounters())
 end
end

"""
getcons(:: MPCCAmpl)
return the set of indices of classical constraints and complementarity cons
"""
function getcons(nlp :: MPCCAmpl)
    Icc = findall(x -> x>0, cvar)
    I   = findall(x -> x==0, cvar)
    return I, Icc
end

function obj(nlp :: MPCCAmpl, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return obj(nlp.mp, x)
end

function grad!(nlp :: MPCCAmpl, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  g .= grad(nlp.mp, x)
  return g
end

function cons_nl!(nlp :: MPCCAmpl, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  I, Icc = getcons(nlp)
  c[1:nlp.meta.ncon] .= cons(nlp, x)[I]
  return c
end

function consG!(nlp :: MPCCAmpl, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consG)
  I, Icc = getcons(nlp)
  c[1:nlp.meta.ncc] .= cons(nlp, x)[Icc]
  return c
end

function consH!(nlp :: MPCCAmpl, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consH)
  I, Icc = getcons(nlp)
  c[1:nlp.meta.ncc] .= x[nlp.cvar[Icc]]
  return c
end

function jac_nl_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncon, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : nlp.meta.nnzj] .= getindex.(I, 1)[:]
  cols[1 : nlp.meta.nnzj] .= getindex.(I, 2)[:]
  return rows, cols
end

function jac_nl_coord!(nlp :: MPCCAmpl, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  I, Icc = getcons(nlp)
  vals[1 : nlp.meta.nnzj] .= (jac(nlp.mp, x)[I,:])[:]
  return vals
end

function jacG_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,j) for i = 1:m, j = 1:n)
  rows[1 : n*m] .= getindex.(I, 1)[:]
  cols[1 : n*m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacG_coord!(nlp :: MPCCAmpl, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  increment!(nlp, :neval_jacG)
  I, Icc = getcons(nlp)
  vals[1 : n*m] .= (jac(nlp.mp, x)[Icc,:])[:]
  return vals
end

function jacH_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  m, n = nlp.meta.ncc, nlp.meta.nvar
  I = ((i,nlp.cvar[Icc[i]]) for i = 1:m)
  rows[1 : m] .= getindex.(I, 1)[:]
  cols[1 : m] .= getindex.(I, 2)[:]
  return rows, cols
end

function jacH_coord!(nlp :: MPCCAmpl, x :: AbstractVector, vals :: AbstractVector)
  m, n = nlp.meta.ncc, nlp.meta.nvar
  increment!(nlp, :neval_jacH)
  #Jx = zeros(m,n)
  #for i = 1:m
  # Jx[i,nlp.cvar[Icc[i]]] = 1.0
  #end
  vals[1 : m] .= ones(m)
  return vals
end
