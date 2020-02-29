##############################################################################
#
# Specialization of AbstractMPCCModel based on automatic differentiation
# Based on AbastractNLPModel:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl
#
# TODO:
# - export
# - doc
# - list of functions not adapted: jth_hprod, jth_hprod!, ghjvprod, ghjvprod!, jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
# - nnzj for G and H
#
##############################################################################

# Methods to be overridden in other packages.
"""
    f = obj(nlp, x)
Evaluate ``f(x)``, the objective function of `mpcc` at `x`.
"""
obj(::AbstractMPCCModel, ::AbstractVector) =
  throw(NotImplementedError("obj"))

"""
    g = grad(nlp, x)
Evaluate ``∇f(x)``, the gradient of the objective function at `x`.
"""
function grad(nlp::AbstractMPCCModel, x::AbstractVector)
  g = similar(x)
  return grad!(nlp, x, g)
end

"""
    g = grad!(nlp, x, g)
Evaluate ``∇f(x)``, the gradient of the objective function at `x` in place.
"""
grad!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("grad!"))

"""
    c = cons(nlp, x)
Evaluate ``c(x)``, the constraints at `x`.
"""
function cons(nlp::AbstractMPCCModel, x::AbstractVector)
  c = similar(x, nlp.meta.ncon)
  return cons!(nlp, x, c)
end

"""
    c = cons!(nlp, x, c)
    Return the vector of the constraints
    lc <= c(x) <= uc,
    lccG <= G(x),
    lccH <= H(x)
"""
function cons!(mod :: AbstractMPCCModel, x :: AbstractVector, c :: AbstractVector)
  if mod.meta.ncc > 0
   G, H = consG(mod, x), consH(mod, x)
  else
   G, H = Float64[], Float64[]
  end

  if mod.meta.ncon > 0
   vnl = cons_nl(mod, x)
  else
   vnl = Float64[]
  end

  c = vcat(vnl, G, H)
  return c
end

"""
    c = viol(nlp, x)
Evaluate ``c(x)``, the constraints at `x`.
"""
function viol(nlp::AbstractMPCCModel, x::AbstractVector)
  c = similar(x, nlp.meta.nvar+nlp.meta.ncon+3*nlp.meta.ncc)
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
function viol!(mod :: AbstractMPCCModel, x :: AbstractVector, c :: AbstractVector)
  n, ncon, ncc = mod.meta.nvar, mod.meta.ncon, mod.meta.ncc
  if ncc > 0
   cG, cH = consG(mod, x), consH(mod, x)

   c[n+ncon+1:n+ncon+ncc]     = max.(mod.meta.lccG - cG, 0.0)
   c[n+ncon+ncc+1:n+ncon+2*ncc] = max.(mod.meta.lccH - cH, 0.0)
   c[n+ncon+2*ncc+1:n+ncon+3*ncc] = max.(cH.*cG, 0.0)
  end

  if mod.meta.ncon > 0
   cx = cons_nl(mod, x)
   c[n+1:n+ncon] = max.(max.(mod.meta.lcon - cx, 0.0),max.(cx - mod.meta.ucon, 0.0))
  end

  c[1:n] = max.(max.(mod.meta.lvar - x, 0.0),max.(x - mod.meta.uvar, 0.0))

  return c
end

"""
Evaluate ``c(x)``, the constraints at `x`.
"""
function cons_nl(mod :: AbstractMPCCModel, x :: AbstractVector)
 c = similar(x, mod.meta.ncon)
 return cons_nl!(mod, x, c)
end

cons_nl!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("cons_nl!"))

"""
Evaluate ``G(x)``, the constraints at `x`.
"""
function consG(mod :: AbstractMPCCModel, x :: AbstractVector)
 c = similar(x, mod.meta.ncc)
 return consG!(mod, x, c)
end

consG!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
       throw(NotImplementedError("consG!"))

"""
Evaluate ``G(x)``, the constraints at `x`.
"""
function consH(mod :: AbstractMPCCModel, x :: AbstractVector)
 c = similar(x, mod.meta.ncc)
 return consH!(mod, x, c)
end

consH!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
              throw(NotImplementedError("consH!"))

jth_con(::AbstractMPCCModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_con"))

function jth_congrad(nlp::AbstractMPCCModel, x::AbstractVector, j::Integer)
  g = Vector{eltype(x)}(undef, nlp.meta.nvar)
  return jth_congrad!(nlp, x, j, g)
end

jth_congrad!(::AbstractMPCCModel, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_congrad!"))

jth_sparse_congrad(::AbstractMPCCModel, ::AbstractVector, ::Integer) =
  throw(NotImplementedError("jth_sparse_congrad"))

"""
    f, c = objcons(nlp, x)
Evaluate ``f(x)`` and ``c(x)`` at `x`.
"""
function objcons(nlp, x)
  f = obj(nlp, x)
  c = nlp.meta.ncon > 0 ? cons(nlp, x) : eltype(x)[]
  return f, c
end

"""
    f = objcons!(nlp, x, c)
Evaluate ``f(x)`` and ``c(x)`` at `x`. `c` is overwritten with the value of ``c(x)``.
"""
function objcons!(nlp, x, c)
  f = obj(nlp, x)
  nlp.meta.ncon > 0 && cons!(nlp, x, c)
  return f, c
end

"""
    f, g = objgrad(nlp, x)
Evaluate ``f(x)`` and ``∇f(x)`` at `x`.
"""
function objgrad(nlp, x)
  f = obj(nlp, x)
  g = grad(nlp, x)
  return f, g
end

"""
    f, g = objgrad!(nlp, x, g)
Evaluate ``f(x)`` and ``∇f(x)`` at `x`. `g` is overwritten with the
value of ``∇f(x)``.
"""
function objgrad!(nlp, x, g)
  f = obj(nlp, x)
  grad!(nlp, x, g)
  return f, g
end

"""
    (rows,cols) = jac_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jac_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzj+2 * nlp.meta.nvar * nlp.meta.ncc)
  cols = Vector{Int}(undef, nlp.meta.nnzj+2 * nlp.meta.nvar * nlp.meta.ncc)
  return jac_structure!(nlp, rows, cols)
end

"""
    jac_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
function jac_structure!(nlp :: AbstractMPCCModel, rows :: AbstractVector{<:Integer}, cols :: AbstractVector{<:Integer})
 m, n = nlp.meta.ncon+2*nlp.meta.ncc, nlp.meta.nvar
 I = ((i,j) for i = 1:m, j = 1:n)
 rows[1 : n*m] .= getindex.(I, 1)[:]
 cols[1 : n*m] .= getindex.(I, 2)[:]
 return rows, cols
end
#I = findall(!iszero, Hx)
#return (getindex.(I, 1), getindex.(I, 2), Hx[I])

"""
    vals = jac_coord!(nlp, x, vals)
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
function jac_coord!(nlp :: AbstractMPCCModel, x :: AbstractVector, vals ::AbstractVector)
 m, n = nlp.meta.ncon+2*nlp.meta.ncc, nlp.meta.nvar
 Jx = vcat(jac_nl(nlp, x),jacG(nlp, x),jacH(nlp, x))
 vals[1 : n*m] .= Jx[:]
 return vals
end
"""
    vals = jac_coord(nlp, x)
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jac_coord(nlp :: AbstractMPCCModel, x :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzj+2*nlp.meta.nvar * nlp.meta.ncc)
  return jac_coord!(nlp, x, vals)
end

"""
    Jx = jac(nlp, x)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]v``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jac(nlp :: AbstractMPCCModel, x :: AbstractVector)
  rows, cols = jac_structure(nlp)
  vals = jac_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.meta.ncon+2*nlp.meta.ncc, nlp.meta.nvar)
end

"""
    (rows,cols) = jac_nl_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jac_nl_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzj)
  cols = Vector{Int}(undef, nlp.meta.nnzj)
  return jac_nl_structure!(nlp, rows, cols)
end

"""
    jac_nl_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
jac_nl_structure!(:: AbstractMPCCModel, :: AbstractVector{<:Integer}, :: AbstractVector{<:Integer}) = throw(NotImplementedError("jac_nl_structure!"))

"""
    vals = jac_nl_coord!(nlp, x, vals)
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
jac_nl_coord!(:: AbstractMPCCModel, :: AbstractVector, ::AbstractVector) = throw(NotImplementedError("jac_nl_coord!"))

"""
    vals = jac_nl_coord(nlp, x)
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jac_nl_coord(nlp :: AbstractMPCCModel, x :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzj)
  return jac_nl_coord!(nlp, x, vals)
end

"""
    Jx = jac_nl(nlp, x)
Evaluate ``∇c(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jac_nl(nlp :: AbstractMPCCModel, x :: AbstractVector)
  rows, cols = jac_nl_structure(nlp)
  vals = jac_nl_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.meta.ncon, nlp.meta.nvar)
end

"""
    (rows,cols) = jacG_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jacG_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nvar * nlp.meta.ncc)
  cols = Vector{Int}(undef, nlp.meta.nvar * nlp.meta.ncc)
  return jacG_structure!(nlp, rows, cols)
end

"""
    jacG_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
jacG_structure!(:: AbstractMPCCModel, :: AbstractVector{<:Integer}, :: AbstractVector{<:Integer}) = throw(NotImplementedError("jacG_structure!"))

"""
    vals = jacG_coord!(nlp, x, vals)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
jacG_coord!(:: AbstractMPCCModel, :: AbstractVector, ::AbstractVector) = throw(NotImplementedError("jacG_coord!"))

"""
    vals = jacG_coord(nlp, x)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jacG_coord(nlp :: AbstractMPCCModel, x :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nvar * nlp.meta.ncc)
  return jacG_coord!(nlp, x, vals)
end
"""
    Jx = jacG(nlp, x)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacG(nlp :: AbstractMPCCModel, x :: AbstractVector)
  rows, cols = jacG_structure(nlp)
  vals = jacG_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.meta.ncc, nlp.meta.nvar)
end

"""
    (rows,cols) = jacH_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jacH_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nvar * nlp.meta.ncc)
  cols = Vector{Int}(undef, nlp.meta.nvar * nlp.meta.ncc)
  return jacH_structure!(nlp, rows, cols)
end

"""
    jacH_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
jacH_structure!(:: AbstractMPCCModel, :: AbstractVector{<:Integer}, :: AbstractVector{<:Integer}) = throw(NotImplementedError("jacH_structure!"))

"""
    vals = jacH_coord!(nlp, x, vals)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
jacH_coord!(:: AbstractMPCCModel, :: AbstractVector, ::AbstractVector) = throw(NotImplementedError("jacH_coord!"))

"""
    vals = jacH_coord(nlp, x)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jacH_coord(nlp :: AbstractMPCCModel, x :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nvar * nlp.meta.ncc)
  return jacH_coord!(nlp, x, vals)
end
"""
    Jx = jacH(nlp, x)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacH(nlp :: AbstractMPCCModel, x :: AbstractVector)
  rows, cols = jacH_structure(nlp)
  vals = jacH_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.meta.ncc, nlp.meta.nvar)
end

"""
    Jv = jprod(nlp, x, v)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]v``, the Jacobian-vector product at `x`.
"""
function jprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.meta.ncon+2*nlp.meta.ncc)
  return jprod!(nlp, x, v, Jv)
end

"""
    Jv = jnlprod(nlp, x, v)
Evaluate ``∇c(x)v``, the Jacobian-vector product at `x`.
"""
function jnlprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.meta.ncon)
  return jnlprod!(nlp, x, v, Jv)
end

"""
    JGv = jGprod(nlp, x, v)
Evaluate ``∇G(x)v``, the Jacobian-vector product at `x`.
"""
function jGprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.meta.ncc)
  return jGprod!(nlp, x, v, Jv)
end

"""
    JHv = jprod(nlp, x, v)
Evaluate ``∇H(x)v``, the Jacobian-vector product at `x`.
"""
function jHprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.meta.ncc)
  return jHprod!(nlp, x, v, Jv)
end

"""
    Jv = jprod!(nlp, x, v, Jv)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]v``, the Jacobian-vector product at `x` in place.
"""
function jprod!(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  ncon, ncc = mod.meta.ncon, mod.meta.ncc
  if ncon > 0
      jnlprod!(nlp, x, v, Jv[1:ncon])
  end
  if ncc > 0
      jGprod!(nlp, x, v, Jv[ncon+1:ncon+ncc])
      jHprod!(nlp, x, v, Jv[ncon+ncc:ncon+2*ncc])
  end
  return Jv
end

"""
 Jv = jnlprod!(nlp, x, v, Jv)
Evaluate ``∇c(x)v``, the Jacobian-vector product at `x` in place.
"""
jnlprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
    throw(NotImplementedError("jnlprod!"))

"""
  JGv = jGprod!(nlp, x, v, Jv)
Evaluate ``∇G(x)v``, the Jacobian-vector product at `x` in place.
"""
jGprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
    throw(NotImplementedError("jGprod!"))

"""
  JHv = jHprod!(nlp, x, v, Jv)
Evaluate ``∇H(x)v``, the Jacobian-vector product at `x` in place.
"""
jHprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
      throw(NotImplementedError("jHprod!"))

"""
    Jtv = jtprod(nlp, x, v, Jtv)
Evaluate ``∇c(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jtprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jtprod!(nlp, x, v, Jtv)
end
"""
    Jtv = jtprodnl(nlp, x, v, Jtv)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]v``, the transposed-Jacobian-vector product at `x`.
"""
function jtprodnl(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jtprodnl!(nlp, x, v, Jtv)
end
"""
    JGtv = jGtprod(nlp, x, v, Jtv)
Evaluate ``∇G(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jGtprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jGtprod!(nlp, x, v, Jtv)
end
"""
    JHtv = jtprod(nlp, x, v, Jtv)
Evaluate ``∇H(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jHtprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jHtprod!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod!(nlp, x, v, Jtv)
Evaluate ``[∇c(x), -∇G(x), -∇H(x)]^Tv``, the transposed-Jacobian-vector product at `x` in place.
"""
function jtprod!(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  ncon, ncc = mod.meta.ncon, mod.meta.ncc
  Jv = zeros(ncon + 2*ncc)
  if ncon > 0
     Jv += jnltprod(nlp, x, v[1:ncon])
  end
  if ncc > 0
      Jv += jGtprod(nlp, x, v[ncon+1:ncon+ncc], Jv)
      Jv += jHtprod(nlp, x, v[ncon+ncc:ncon+2*ncc], Jv)
  end
  return Jv
end

"""
 Jtv = jnltprod!(nlp, x, v, Jtv)
Evaluate ``∇c(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
"""
jnltprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
    throw(NotImplementedError("jnltprod!"))
"""
 JGtv = jGtprod!(nlp, x, v, Jtv)
 Evaluate ``∇G(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
"""
jGtprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
    throw(NotImplementedError("jGtprod!"))
"""
JHtv = jHtprod!(nlp, x, v, Jtv)
Evaluate ``∇H(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
"""
jHtprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
      throw(NotImplementedError("jHtprod!"))

"""
    J = jac_op(nlp, x)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_op(nlp :: AbstractMPCCModel, x :: AbstractVector)
  prod = @closure v -> jprod(nlp, x, v)
  ctprod = @closure v -> jtprod(nlp, x, v)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

"""
    J = jac_op!(nlp, x, Jv, Jtv)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_op!(nlp :: AbstractMPCCModel, x :: AbstractVector,
                 Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = @closure v -> jprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jtprod!(nlp, x, v, Jtv)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

"""
    J = jacG_op(nlp, x)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jacG_op(nlp :: AbstractMPCCModel, x :: AbstractVector)
  prod = @closure v -> jGprod(nlp, x, v)
  ctprod = @closure v -> jGtprod(nlp, x, v)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

"""
    J = jacG_op!(nlp, x, Jv, Jtv)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jacG_op!(nlp :: AbstractMPCCModel, x :: AbstractVector,
                 Jv :: AbstractVector, Jtv :: AbstractVector)
  prod = @closure v -> jGprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jGtprod!(nlp, x, v, Jtv)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

"""
    J = jacH_op(nlp, x)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jacH_op(nlp :: AbstractMPCCModel, x :: AbstractVector)
  prod = @closure v -> jHprod(nlp, x, v)
  ctprod = @closure v -> jHtprod(nlp, x, v)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

"""
    J = jacH_op!(nlp, x, Jv, Jtv)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jacH_op!(nlp :: AbstractMPCCModel, x :: AbstractVector,
                 Jv :: AbstractVector, Jtv :: AbstractVector)
  prod   = @closure v -> jHprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jHtprod!(nlp, x, v, Jtv)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

function jth_hprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, j::Integer)
  hv = Vector{eltype(x)}(undef, nlp.meta.nvar)
  return jth_hprod!(nlp, x, v, j, hv)
end

jth_hprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::Integer, ::AbstractVector) =
  throw(NotImplementedError("jth_hprod!"))

function ghjvprod(nlp::AbstractMPCCModel, x::AbstractVector, g::AbstractVector, v::AbstractVector)
  gHv = Vector{eltype(x)}(undef, nlp.meta.ncon)
  return ghjvprod!(nlp, x, g, v, gHv)
end

ghjvprod!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("ghjvprod!"))

"""
    (rows,cols) = hess_structure(nlp)
Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hess_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  hess_structure!(nlp, rows, cols)
end

"""
    hess_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hess_structure!(:: AbstractMPCCModel, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}) = throw(NotImplementedError("hess_structure!"))

"""
    vals = hess_coord!(nlp, x, vals; obj_weight=1.0)
Evaluate the objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
OBJECTIVE_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
function hess_coord!(nlp :: AbstractMPCCModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real=1.0)
  hess_coord!(nlp, x, zeros(nlp.meta.ncon), vals, obj_weight=obj_weight)
end

"""
    vals = hess_coord!(nlp, x, y, vals; obj_weight=1.0)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
hess_coord!(nlp:: AbstractMPCCModel, :: AbstractVector, :: AbstractVector, ::AbstractVector; obj_weight :: Real=1.0) = throw(NotImplementedError("hess_coord!"))

"""
    vals = hess_coord(nlp, x; obj_weight=1.0)
Evaluate the objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
OBJECTIVE_HESSIAN.
Only the lower triangle is returned.
"""
function hess_coord(nlp :: AbstractMPCCModel, x :: AbstractVector; obj_weight::Real=one(eltype(x)))
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  return hess_coord!(nlp, x, vals; obj_weight=obj_weight)
end

"""
    vals = hess_coord(nlp, x, y; obj_weight=1.0)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hess_coord(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector; obj_weight::Real=one(eltype(x)))
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  return hess_coord!(nlp, x, y, vals; obj_weight=obj_weight)
end

"""
    Hx = hess(nlp, x; obj_weight=1.0)
Evaluate the objective Hessian at `x` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
OBJECTIVE_HESSIAN.
Only the lower triangle is returned.
"""
function hess(nlp::AbstractMPCCModel, x::AbstractVector; obj_weight::Real=one(eltype(x)))
  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, obj_weight=obj_weight)
  return tril(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar))
end

"""
    Hx = hess(nlp, x, y; obj_weight=1.0)
Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hess(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector; obj_weight::Real=one(eltype(x)))
  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, y, obj_weight=obj_weight)
  return tril(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar))
end

"""
    (rows,cols) = hessG_structure(nlp)
Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hessG_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  hessG_structure!(nlp, rows, cols)
end

"""
    hessG_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hessG_structure!(:: AbstractMPCCModel, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}) = throw(NotImplementedError("hessG_structure!"))

"""
    vals = hess_coord!(nlp, x, y, vals)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
hessG_coord!(nlp:: AbstractMPCCModel, :: AbstractVector, :: AbstractVector, ::AbstractVector) = throw(NotImplementedError("hessG_coord!"))

"""
    vals = hessG_coord(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessG_coord(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  return hessG_coord!(nlp, x, y, vals)
end

"""
    Hx = hessG(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessG(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
  rows, cols = hessG_structure(nlp)
  vals = hessG_coord(nlp, x, y)
  return tril(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar))
end

"""
    (rows,cols) = hessG_structure(nlp)
Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hessH_structure(nlp :: AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  hessH_structure!(nlp, rows, cols)
end

"""
    hessH_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hessH_structure!(:: AbstractMPCCModel, ::AbstractVector{<: Integer}, ::AbstractVector{<: Integer}) = throw(NotImplementedError("hessH_structure!"))

"""
    vals = hessH_coord!(nlp, x, y, vals)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
hessH_coord!(nlp:: AbstractMPCCModel, :: AbstractVector, :: AbstractVector, ::AbstractVector) = throw(NotImplementedError("hessH_coord!"))

"""
    vals = hessG_coord(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessH_coord(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  return hessH_coord!(nlp, x, y, vals)
end

"""
    Hx = hessH(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessH(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
  rows, cols = hessH_structure(nlp)
  vals = hessH_coord(nlp, x, y)
  return tril(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar))
end

"""
    Hv = hprod(nlp, x, v; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector; obj_weight::Real=one(eltype(x)))
  Hv = similar(x)
  return hprod!(nlp, x, v, Hv; obj_weight=obj_weight)
end

"""
    Hv = hprod(nlp, x, y, v; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
function hprod(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector, v::AbstractVector; obj_weight::Real=one(eltype(x)))
  Hv = similar(x)
  return hprod!(nlp, x, y, v, Hv; obj_weight=obj_weight)
end

"""
    Hv = hprod!(nlp, x, v, Hv; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hprod!(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight :: Real=1.0)
  hprod!(nlp, x, zeros(nlp.meta.ncon), v, Hv, obj_weight=obj_weight)
end

"""
    Hv = hprod!(nlp, x, y, v, Hv; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
hprod!(nlp::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector; obj_weight :: Real=1.0) =
  throw(NotImplementedError("hprod!"))

"""
 Hv = hGprod!(nlp, x, v, Hv; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hGprod!(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
 hGprod!(nlp, x, zeros(nlp.meta.ncon), v, Hv)
end

"""
Hv = hGprod!(nlp, x, y, v, Hv; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
hGprod!(nlp::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
throw(NotImplementedError("hGprod!"))

"""
 Hv = hHprod!(nlp, x, v, Hv; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hHprod!(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector)
 hHprod!(nlp, x, zeros(nlp.meta.ncon), v, Hv)
end

"""
Hv = hHprod!(nlp, x, y, v, Hv; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
hHprod!(nlp::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
throw(NotImplementedError("hHprod!"))

"""
    H = hess_op(nlp, x; obj_weight=1.0)
Return the objective Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
OBJECTIVE_HESSIAN.
"""
function hess_op(nlp :: AbstractMPCCModel, x :: AbstractVector; obj_weight::Real=one(eltype(x)))
  prod = @closure v -> hprod(nlp, x, v; obj_weight=obj_weight)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hess_op(nlp, x, y; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
LAGRANGIAN_HESSIAN.
"""
function hess_op(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector; obj_weight::Real=one(eltype(x)))
  prod = @closure v -> hprod(nlp, x, y, v; obj_weight=obj_weight)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hessG_op(nlp, x, y; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
LAGRANGIAN_HESSIAN.
"""
function hessG_op(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector)
  prod = @closure v -> hGprod(nlp, x, y, v)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hessH_op(nlp, x, y; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
LAGRANGIAN_HESSIAN.
"""
function hessH_op(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector)
  prod = @closure v -> hHprod(nlp, x, y, v)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hess_op!(nlp, x, Hv; obj_weight=1.0)
Return the objective Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
OBJECTIVE_HESSIAN.
"""
function hess_op!(nlp :: AbstractMPCCModel, x :: AbstractVector, Hv :: AbstractVector; obj_weight::Real=one(eltype(x)))
  prod = @closure v -> hprod!(nlp, x, v, Hv; obj_weight=obj_weight)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hess_op!(nlp, x, y, Hv; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
LAGRANGIAN_HESSIAN.
"""
function hess_op!(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector, Hv :: AbstractVector; obj_weight::Real=one(eltype(x)))
  prod = @closure v -> hprod!(nlp, x, y, v, Hv; obj_weight=obj_weight)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hessG_op!(nlp, x, y, Hv; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
LAGRANGIAN_HESSIAN.
"""
function hessG_op!(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector, Hv :: AbstractVector)
  prod = @closure v -> hGprod!(nlp, x, y, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hessH_op!(nlp, x, y, Hv; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.  The linear operator H
represents
LAGRANGIAN_HESSIAN.
"""
function hessH_op!(nlp :: AbstractMPCCModel, x :: AbstractVector, y :: AbstractVector, Hv :: AbstractVector)
  prod = @closure v -> hHprod!(nlp, x, y, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end
