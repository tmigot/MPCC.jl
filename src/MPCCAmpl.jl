"""
Definit le type MPCC :
min f(x)
l <= x <= u
lb <= c(x) <= ub

with an array of Int cvar of the same size as c(x).
cvar[i] is > 0 if c_i(x) is involved in a complementarity constraint:
(lb_i = c_i(x) or c_i(x) = ub_i) OR x[cvar[i]] = 0.
"""
mutable struct MPCCAmpl <: AbstractMPCCModel

 mp   :: AbstractNLPModel
 cvar :: AbstractVector{Int64}

 meta :: MPCCModelMeta

 counters :: MPCCCounters

 function MPCCAmpl(mp   :: AbstractNLPModel,
                   cvar :: AbstractVector{Int64},
                   ncc  :: Int64;
                   x0   :: Vector = mp.meta.x0)

  n    = length(x0)
  ncon = mp.meta.ncon

  Icc = findall(x -> x>0, cvar)
  I   = findall(x -> x==0, cvar)

  meta = MPCCModelMeta(n, x0 = x0, ncc = ncc, lccG = mp.meta.lcon[Icc], lccH = mp.meta.lvar[cvar[Icc]],
                              lvar = mp.meta.lvar, uvar = mp.meta.uvar,
                              ncon = ncon,
                              lcon = mp.meta.lcon[I], ucon = mp.meta.ucon[I])

  return new(mp, cvar, meta, MPCCCounters())
 end
end
