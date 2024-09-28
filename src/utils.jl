"""
    c = viol(nlp, x)
Evaluate ``c(x)``, the constraints at `x`.
"""
function viol(nlp::AbstractMPCCModel, x::AbstractVector)
  c = similar(x, nlp.meta.nvar + nlp.meta.ncon + 3 * nlp.cc_meta.ncc)
  return viol!(nlp, x, c)
end

"""
    c = viol!(nlp, x, c)
    Return the vector of the constraints
    lx <= x <= ux
    lc <= c(x) <= uc,
    lccG <= G(x),
    lccH <= H(x),
    G(x) .* H(x) <= 0
"""
function viol!(mod::AbstractMPCCModel, x::AbstractVector, c::AbstractVector)
  n, ncon, ncc = mod.meta.nvar, mod.meta.ncon, mod.meta.ncc
  if ncc > 0
    cG, cH = consG(mod, x), consH(mod, x)

    c[n+ncon+1:n+ncon+ncc] = max.(mod.meta.lccG - cG, 0.0)
    c[n+ncon+ncc+1:n+ncon+2*ncc] = max.(mod.meta.lccH - cH, 0.0)
    c[n+ncon+2*ncc+1:n+ncon+3*ncc] = max.(cH .* cG, 0.0)
  end

  if mod.meta.ncon > 0
    cx = cons_nl(mod, x)
    c[n+1:n+ncon] = max.(max.(mod.meta.lcon - cx, 0.0), max.(cx - mod.meta.ucon, 0.0))
  end

  c[1:n] = max.(max.(mod.meta.lvar - x, 0.0), max.(x - mod.meta.uvar, 0.0))
  return c
end
