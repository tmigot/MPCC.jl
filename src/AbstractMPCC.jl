for field in [:G, :H]

  name = Symbol("cons", field)
  namein = Symbol("cons", field, "!")

  for clinnln in [:lin, :nln]
    nameclinnln = clinnln == :lin ? :linear : :nonlinear
    namelin = Symbol("cons", field, "_", clinnln)
    namelinin = Symbol("cons", field, "_", clinnln, "!")
    """
      c = $($namelin)(nlp, x, c)

    Evaluate the $($nameclinnln) constraints of $($field) at `x`.
    """
    function $namelin(nlp::AbstractMPCCModel, x::AbstractVector)
        c = similar(x, nlp.cc_meta.ncc)
        return $namelinin(nlp, x, c)
    end
    """
      c = $($namelinin)(nlp, x, c)

    Evaluate the $($nameclinnln) constraints at of $($field) `x` in place.
    """
    function $namein end
  end

  """
    c = $($name)(nlp, x, c)

  Evaluate the constraints of $($name) at `x`.
  """
  function $name(nlp::AbstractMPCCModel, x::AbstractVector)
      c = similar(x, nlp.cc_meta.ncc)
      return $namein(nlp, x, c)
  end
  """
    c = $($name)(nlp, x, c)

  Evaluate the constraints of $($name) at `x` in place.
  """
  function $namein end
end

"""
    (rows,cols) = jacG_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jacG_structure(nlp::AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.cc_meta.nnzjG)
  cols = Vector{Int}(undef, nlp.cc_meta.nnzjG)
  return jacG_structure!(nlp, rows, cols)
end

"""
    jacG_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
jacG_structure!(
  ::AbstractMPCCModel,
  ::AbstractVector{<:Integer},
  ::AbstractVector{<:Integer},
) = throw(NotImplementedError("jacG_structure!"))

"""
    vals = jacG_coord!(nlp, x, vals)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
jacG_coord!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jacG_coord!"))

"""
    vals = jacG_coord(nlp, x)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jacG_coord(nlp::AbstractMPCCModel, x::AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.cc_meta.nnzjG)
  return jacG_coord!(nlp, x, vals)
end
"""
    Jx = jacG(nlp, x)
Evaluate ``∇G(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacG(nlp::AbstractMPCCModel, x::AbstractVector)
  rows, cols = jacG_structure(nlp)
  vals = jacG_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.cc_meta.ncc, nlp.meta.nvar)
end

"""
    (rows,cols) = jacH_structure(nlp)
Return the structure of the constraint's Jacobian in sparse coordinate format.
"""
function jacH_structure(nlp::AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.cc_meta.nnzjH)
  cols = Vector{Int}(undef, nlp.cc_meta.nnzjH)
  return jacH_structure!(nlp, rows, cols)
end

"""
    jacH_structure!(nlp, rows, cols)
Return the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
jacH_structure!(
  ::AbstractMPCCModel,
  ::AbstractVector{<:Integer},
  ::AbstractVector{<:Integer},
) = throw(NotImplementedError("jacH_structure!"))

"""
    vals = jacH_coord!(nlp, x, vals)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` in sparse coordinate format,
rewriting `vals`.
"""
jacH_coord!(::AbstractMPCCModel, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("jacH_coord!"))

"""
    vals = jacH_coord(nlp, x)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` in sparse coordinate format.
"""
function jacH_coord(nlp::AbstractMPCCModel, x::AbstractVector)
  vals = Vector{eltype(x)}(undef, nlp.cc_meta.nnzjH)
  return jacH_coord!(nlp, x, vals)
end
"""
    Jx = jacH(nlp, x)
Evaluate ``∇H(x)``, the constraint's Jacobian at `x` as a sparse matrix.
"""
function jacH(nlp::AbstractMPCCModel, x::AbstractVector)
  rows, cols = jacH_structure(nlp)
  vals = jacH_coord(nlp, x)
  return sparse(rows, cols, vals, nlp.cc_meta.ncc, nlp.meta.nvar)
end

"""
    JGv = jGprod(nlp, x, v)
Evaluate ``∇G(x)v``, the Jacobian-vector product at `x`.
"""
function jGprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.cc_meta.ncc)
  return jGprod!(nlp, x, v, Jv)
end

"""
    JHv = jHprod(nlp, x, v)
Evaluate ``∇H(x)v``, the Jacobian-vector product at `x`.
"""
function jHprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jv = similar(v, nlp.cc_meta.ncc)
  return jHprod!(nlp, x, v, Jv)
end

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
    JGtv = jGtprod(nlp, x, v, Jtv)
Evaluate ``∇G(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jGtprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jGtprod!(nlp, x, v, Jtv)
end

"""
    JHtv = jHtprod(nlp, x, v, Jtv)
Evaluate ``∇H(x)^Tv``, the transposed-Jacobian-vector product at `x`.
"""
function jHtprod(nlp::AbstractMPCCModel, x::AbstractVector, v::AbstractVector)
  Jtv = similar(x)
  return jHtprod!(nlp, x, v, Jtv)
end

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
    J = jacG_op(nlp, x)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jacG_op(nlp::AbstractMPCCModel, x::AbstractVector)
  prod = @closure v -> jGprod(nlp, x, v)
  ctprod = @closure v -> jGtprod(nlp, x, v)
  return LinearOperator{eltype(x)}(
    nlp.meta.ncon,
    nlp.meta.nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    J = jacG_op!(nlp, x, Jv, Jtv)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jacG_op!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  prod = @closure v -> jGprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jGtprod!(nlp, x, v, Jtv)
  return LinearOperator{eltype(x)}(
    nlp.meta.ncon,
    nlp.meta.nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    J = jacH_op(nlp, x)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jacH_op(nlp::AbstractMPCCModel, x::AbstractVector)
  prod = @closure v -> jHprod(nlp, x, v)
  ctprod = @closure v -> jHtprod(nlp, x, v)
  return LinearOperator{eltype(x)}(
    nlp.meta.ncon,
    nlp.meta.nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    J = jacH_op!(nlp, x, Jv, Jtv)
Return the Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jacH_op!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  prod = @closure v -> jHprod!(nlp, x, v, Jv)
  ctprod = @closure v -> jHtprod!(nlp, x, v, Jtv)
  return LinearOperator{eltype(x)}(
    nlp.meta.ncon,
    nlp.meta.nvar,
    false,
    false,
    prod,
    ctprod,
    ctprod,
  )
end

"""
    (rows,cols) = hessG_structure(nlp)
Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hessG_structure(nlp::AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  return hessG_structure!(nlp, rows, cols)
end

"""
    hessG_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hessG_structure!(
  ::AbstractMPCCModel,
  ::AbstractVector{<:Integer},
  ::AbstractVector{<:Integer},
) = throw(NotImplementedError("hessG_structure!"))

"""
    vals = hess_coord!(nlp, x, y, vals)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
hessG_coord!(nlp::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("hessG_coord!"))

"""
    vals = hessG_coord(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessG_coord(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
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
function hessH_structure(nlp::AbstractMPCCModel)
  rows = Vector{Int}(undef, nlp.meta.nnzh)
  cols = Vector{Int}(undef, nlp.meta.nnzh)
  return hessH_structure!(nlp, rows, cols)
end

"""
    hessH_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
hessH_structure!(
  ::AbstractMPCCModel,
  ::AbstractVector{<:Integer},
  ::AbstractVector{<:Integer},
) = throw(NotImplementedError("hessH_structure!"))

"""
    vals = hessH_coord!(nlp, x, y, vals)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN, rewriting `vals`.
Only the lower triangle is returned.
"""
hessH_coord!(nlp::AbstractMPCCModel, ::AbstractVector, ::AbstractVector, ::AbstractVector) =
  throw(NotImplementedError("hessH_coord!"))

"""
    vals = hessG_coord(nlp, x, y)
Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
LAGRANGIAN_HESSIAN.
Only the lower triangle is returned.
"""
function hessH_coord(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
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
    Hv = hGprod(nlp, x, y, v)
Evaluate the product of the G Hessian at `(x,y)` with the vector `v`.
"""
function hGprod(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
)
  Hv = similar(x)
  return hGprod!(nlp, x, y, v, Hv)
end

"""
    Hv = hHprod(nlp, x, y, v)
Evaluate the product of the G Hessian at `(x,y)` with the vector `v`.
"""
function hHprod(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
)
  Hv = similar(x)
  return hHprod!(nlp, x, y, v, Hv)
end

"""
 Hv = hGprod!(nlp, x, v, Hv; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hGprod!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  return hGprod!(nlp, x, zeros(nlp.meta.ncon), v, Hv)
end

"""
Hv = hGprod!(nlp, x, y, v, Hv; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
hGprod!(
  nlp::AbstractMPCCModel,
  ::AbstractVector,
  ::AbstractVector,
  ::AbstractVector,
  ::AbstractVector,
) = throw(NotImplementedError("hGprod!"))

"""
 Hv = hHprod!(nlp, x, v, Hv; obj_weight=1.0)
Evaluate the product of the objective Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the objective Hessian is
OBJECTIVE_HESSIAN.
"""
function hHprod!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  return hHprod!(nlp, x, zeros(nlp.meta.ncon), v, Hv)
end

"""
Hv = hHprod!(nlp, x, y, v, Hv; obj_weight=1.0)
Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`, where the Lagrangian Hessian is
LAGRANGIAN_HESSIAN.
"""
hHprod!(
  nlp::AbstractMPCCModel,
  ::AbstractVector,
  ::AbstractVector,
  ::AbstractVector,
  ::AbstractVector,
) = throw(NotImplementedError("hHprod!"))

"""
    H = hessG_op(nlp, x, y; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
LAGRANGIAN_HESSIAN.
"""
function hessG_op(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
  prod = @closure v -> hGprod(nlp, x, y, v)
  return LinearOperator{eltype(x)}(
    nlp.meta.nvar,
    nlp.meta.nvar,
    true,
    true,
    prod,
    prod,
    prod,
  )
end

"""
    H = hessH_op(nlp, x, y; obj_weight=1.0)
Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`. The linear operator H represents
LAGRANGIAN_HESSIAN.
"""
function hessH_op(nlp::AbstractMPCCModel, x::AbstractVector, y::AbstractVector)
  prod = @closure v -> hHprod(nlp, x, y, v)
  return LinearOperator{eltype(x)}(
    nlp.meta.nvar,
    nlp.meta.nvar,
    true,
    true,
    prod,
    prod,
    prod,
  )
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
function hessG_op!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector,
)
  prod = @closure v -> hGprod!(nlp, x, y, v, Hv)
  return LinearOperator{eltype(x)}(
    nlp.meta.nvar,
    nlp.meta.nvar,
    true,
    true,
    prod,
    prod,
    prod,
  )
end

function hess_coord(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck (nlp.meta.ncon + 2 * nlp.cc_meta.ncc) y
  vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  return hess_coord!(nlp, x, y, vals; obj_weight = obj_weight)
end

function hess(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x
  @lencheck (nlp.meta.ncon + 2 * nlp.cc_meta.ncc) y
  rows, cols = hess_structure(nlp)
  vals = hess_coord(nlp, x, y, obj_weight = obj_weight)
  Symmetric(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar), :L)
end

function hprod(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x v
  @lencheck (nlp.meta.ncon + 2 * nlp.cc_meta.ncc) y
  Hv = similar(x)
  return hprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
end

function hess_op(
  nlp::AbstractMPCCModel{T,S},
  x::AbstractVector{T},
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T,S}
  @lencheck nlp.meta.nvar x
  @lencheck (nlp.meta.ncon + 2 * nlp.cc_meta.ncc) y
  Hv = S(undef, nlp.meta.nvar)
  return hess_op!(nlp, x, y, Hv, obj_weight = obj_weight)
end

function hess_op!(
  nlp::AbstractMPCCModel,
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(eltype(x)),
)
  @lencheck nlp.meta.nvar x Hv
  @lencheck (nlp.meta.ncon + 2 * nlp.cc_meta.ncc) y
  prod! = @closure (res, v, α, β) -> begin
    hprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
    if β == 0
      @. res = α * Hv
    else
      @. res = α * Hv + β * res
    end
    return res
  end
  return LinearOperator{eltype(x)}(
    nlp.meta.nvar,
    nlp.meta.nvar,
    true,
    true,
    prod!,
    prod!,
    prod!,
  )
end
