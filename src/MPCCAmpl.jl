"""
Definit le type MPCC :
min f(x)
l <= x <= u
lb <= c(x) <= ub

with an array of integer "cvar" of the same size as c(x).
cvar[i] is > 0 if c_i(x) is involved in a complementarity constraint:
(lb_i = c_i(x) or c_i(x) = ub_i) OR x[cvar[i]] = 0.

Note: cons(nlp, x) call twice cons(nlp.mp, x) [TO FIX - idem jac, hess]

TODO: connect with https://github.com/tmigot/AmplNLReader.jl/blob/master/src/ampl_cc_model.jl
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
    Icc = findall(x -> x>0, nlp.cvar)
    I   = findall(x -> x==0, nlp.cvar)
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
  if nlp.meta.ncon > 0
   c[1:nlp.meta.ncon] .= cons(nlp, x)[I]
  else
    c = zeros(0)
  end
  return c
end

function consG!(nlp :: MPCCAmpl, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consG)
  I, Icc = getcons(nlp)
  if nlp.meta.ncc > 0
   c[1:nlp.meta.ncc] .= cons(nlp, x)[Icc]
  else
    c = zeros(0)
  end
  return c
end

function consH!(nlp :: MPCCAmpl, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_consH)
  I, Icc = getcons(nlp)
  if nlp.meta.ncc > 0
   c[1:nlp.meta.ncc] .= x[nlp.cvar[Icc]]
  else
    c = zeros(0)
  end
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
  if nlp.meta.ncc > 0
   vals[1 : n*m] .= (jac(nlp.mp, x)[Icc,:])[:]
  else
   vals = zeros(0)
  end
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

function jnlprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv[1:nlp.meta.ncon] .= jac_nl(nlp, x) * v
  return Jv
end

function jnltprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv[1:nlp.meta.nvar] .= jac_nl(nlp, x)' * v
  return Jtv
end

function jGprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jGprod)
  Jv[1:nlp.meta.ncc] .= jacG(nlp, x) * v
  return Jv
end

function jGtprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jGtprod)
  Jtv[1:nlp.meta.nvar] .= jacG(nlp, x)' * v
  return Jtv
end

function jHprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jHprod)
  Jv[1:nlp.meta.ncc] .= jacH(nlp, x) * v
  return Jv
end

function jHtprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jHtprod)
  Jtv[1:nlp.meta.nvar] .= jacH(nlp, x)' * v
  return Jtv
end

function hess_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: MPCCAmpl, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  Hx = hess(nlp.mp, x, obj_weight = obj_weight)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hess_coord!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hess)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  I, Icc = getcons(nlp)
  ylong  = zeros(ncon + ncc)
  ylong[I] = y[1:ncon]
  Hx = hess(nlp.mp, x, ylong, obj_weight = obj_weight) - hessG(nlp, x, y[ncon+1:ncon+ncc]) - hessH(nlp, x, y[ncon+ncc+1:ncon+2*ncc])
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessG(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessG)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  I, Icc = getcons(nlp)
  ylong  = zeros(ncon + ncc)
  ylong[Icc] = y[1:ncc]
  Hx = hess(nlp.mp, x, ylong, obj_weight = 0.0)
  return tril(Hx)
end

function hessG_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessG_coord!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_hessG)
  Hx = hessG(nlp, x, y)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hessH(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector)
  increment!(nlp, :neval_hessH)
  Hx = zeros(nlp.meta.nvar, nlp.meta.nvar)
  return tril(Hx)
end

function hessH_structure!(nlp :: MPCCAmpl, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows[1 : nlp.meta.nnzh] .= getindex.(I, 1)
  cols[1 : nlp.meta.nnzh] .= getindex.(I, 2)
  return rows, cols
end

function hessH_coord!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_hess)
  Hx = zeros(nlp.meta.nvar, nlp.meta.nvar)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function hprod!(nlp :: MPCCAmpl, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  Hv .= hprod(nlp.mp, x, v, obj_weight = obj_weight)
  return Hv
end

function hprod!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  increment!(nlp, :neval_hprod)
  ncon, ncc = nlp.meta.ncon, nlp.meta.ncc
  Hv .= hess(nlp, x, y) * v
  return Hv
end

function hGprod!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  increment!(nlp, :neval_hGprod)
  Hv .= hessG(nlp, x, y) * v
  return Hv
end

function hHprod!(nlp :: MPCCAmpl, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  increment!(nlp, :neval_hHprod)
  Hv .= hessH(nlp, x, y) * v
  return Hv
end
