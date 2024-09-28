for field in [:G, :H]

  name = Symbol("cons", field)
  namein = Symbol("cons", field, "!")

  for clinnln in [:lin, :nln]
    nameclinnln = clinnln == :lin ? "linear" : "nonlinear"
    nccnlin = Symbol("n", field, clinnln)

    namelin = Symbol("cons", field, "_", clinnln)
    namelinin = Symbol("cons", field, "_", clinnln, "!")

    nameJstructlin = Symbol("jac", field, "_", clinnln, "_structure")
    nameJstructlinin = Symbol("jac", field, "_", clinnln, "_structure!")
    nccnlinnnzj = Symbol(field, clinnln, "_nnzj")
    nameJcoordlin = Symbol("jac", field, "_", clinnln, "_coord")
    nameJcoordlinin = Symbol("jac", field, "_", clinnln, "_coord!")

    nameJprodlin = Symbol("j", field, "prod", clinnln)
    nameJprodlinin = Symbol("j", field, "prod", clinnln, "!")
    nameJtprodlin = Symbol("j", field, "tprod", clinnln)
    nameJtprodlinin = Symbol("j", field, "tprod", clinnln, "!")

    nameJoplin = Symbol("jac", field, "_", clinnln, "_op")
    nameJoplinin = Symbol("jac", field, "_", clinnln, "_op!")

    @eval begin
      @doc """
          c = $($namelin)(nlp, x, c)

      Evaluate the $($nameclinnln) constraints at `x`.
      """
      function $namelin(nlp::AbstractMPCCModel, x::AbstractVector)
        c = similar(x, nlp.cc_meta.$nccnlin)
        return $namelinin(nlp, x, c)
      end
      @doc """
          c = $($namelinin)(nlp, x, c)

      Evaluate the $($nameclinnln) constraints at `x` in place.
      """
      function $namelinin end

      """
          (rows,cols) = $($nameJstructlin)(nlp)

      Return the structure of the $($nameclinnln) constraints Jacobian in sparse coordinate format.
      """
      function $nameJstructlin(nlp::AbstractMPCCModel)
        rows = Vector{Int}(undef, nlp.cc_meta.$nccnlinnnzj)
        cols = Vector{Int}(undef, nlp.cc_meta.$nccnlinnnzj)
        $nameJstructlinin(nlp, rows, cols)
      end

      """
          $($nameJstructlinin)(nlp, rows, cols)

      Return the structure of the $($nameclinnln) constraints Jacobian in sparse coordinate format in place.
      """
      function $nameJstructlinin end

      """
          vals = $($nameJcoordlinin)(nlp, x, vals)

      Evaluate ``J(x)``, the $($nameclinnln) constraints Jacobian at `x` in sparse coordinate format,
      overwriting `vals`.
      """
      function $nameJcoordlinin end

      """
          vals = $($nameJcoordlin)(nlp, x)

      Evaluate ``J(x)``, the $($nameclinnln) constraints Jacobian at `x` in sparse coordinate format.
      """
      function $nameJcoordlin(nlp::AbstractMPCCModel{T,S}, x::AbstractVector) where {T,S}
        @lencheck nlp.meta.nvar x
        vals = S(undef, nlp.cc_meta.$nccnlinnnzj)
        return $nameJcoordlinin(nlp, x, vals)
      end

      """
          Jv = $($nameJprodlin)(nlp, x, v)

      Evaluate ``J(x)v``, the $($nameclinnln) Jacobian-vector product at `x`.
      """
      function $nameJprodlin(
        nlp::AbstractMPCCModel{T,S},
        x::AbstractVector,
        v::AbstractVector,
      ) where {T,S}
        @lencheck nlp.meta.nvar x v
        Jv = S(undef, nlp.cc_meta.$nccnlin)
        return $nameJprodlinin(nlp, x, v, Jv)
      end

      """
          Jv = $($nameJprodlinin)(nlp, x, v, Jv)

      Evaluate ``J(x)v``, the $($nameclinnln) Jacobian-vector product at `x` in place.
      """
      function $nameJprodlinin end

      """
          Jtv = $($nameJtprodlin)(nlp, x, v)

      Evaluate ``J(x)^Tv``, the $($nameclinnln) transposed-Jacobian-vector product at `x`.
      """
      function $nameJtprodlin(
        nlp::AbstractMPCCModel{T,S},
        x::AbstractVector,
        v::AbstractVector,
      ) where {T,S}
        @lencheck nlp.meta.nvar x
        @lencheck nlp.cc_meta.$nccnlin v
        Jtv = S(undef, nlp.meta.nvar)
        return $nameJtprodlinin(nlp, x, v, Jtv)
      end

      """
          Jtv = $($nameJtprodlinin)(nlp, x, v, Jtv)

      Evaluate ``J(x)^Tv``, the $($nameclinnln) transposed-Jacobian-vector product at `x` in place.
      """
      function $nameJtprodlinin end

      """
          J = $($nameJoplin)(nlp, x)

      Return the $($nameclinnln) Jacobian at `x` as a linear operator.
      The resulting object may be used as if it were a matrix, e.g., `J * v` or
      `J' * v`.
      """
      function $nameJoplin(nlp::AbstractMPCCModel{T,S}, x::AbstractVector) where {T,S}
        @lencheck nlp.meta.nvar x
        Jv = S(undef, nlp.cc_meta.$nccnlin)
        Jtv = S(undef, nlp.meta.nvar)
        return $nameJoplinin(nlp, x, Jv, Jtv)
      end

      """
          J = $($nameJoplinin)(nlp, x, Jv, Jtv)

      Return the $($nameclinnln) Jacobian at `x` as a linear operator.
      The resulting object may be used as if it were a matrix, e.g., `J * v` or
      `J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
      operations.
      """
      function $nameJoplinin(
        nlp::AbstractMPCCModel{T,S},
        x::AbstractVector{T},
        Jv::AbstractVector,
        Jtv::AbstractVector,
      ) where {T,S}
        @lencheck nlp.meta.nvar x Jtv
        @lencheck nlp.cc_meta.$nccnlin Jv
        prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
          $nameJprodlinin(nlp, x, v, Jv)
          if β == 0
            @. res = α * Jv
          else
            @. res = α * Jv + β * res
          end
          return res
        end
        ctprod! = @closure (res, v, α, β) -> begin
          $nameJtprodlinin(nlp, x, v, Jtv)
          if β == 0
            @. res = α * Jtv
          else
            @. res = α * Jtv + β * res
          end
          return res
        end
        return LinearOperator{T}(
          nlp.cc_meta.$nccnlin,
          nlp.meta.nvar,
          false,
          false,
          prod!,
          ctprod!,
          ctprod!,
        )
      end

    end
  end

  namelin = Symbol("cons", field, "_lin!")
  namenln = Symbol("cons", field, "_nln!")

  ncccount = Symbol("neval_cons", field)

  nccnlin = Symbol("n", field, "lin")
  ncclin = Symbol(field, "lin")
  nccnnln = Symbol("n", field, "nln")
  nccnln = Symbol(field, "nln")

  nameJstruct = Symbol("jac", field, "_structure")
  nameJstructin = Symbol("jac", field, "_structure!")
  nameJcoord = Symbol("jac", field, "_coord")
  nameJcoordin = Symbol("jac", field, "_coord!")

  namelinJstruct = Symbol("jac", field, "_lin_structure!")
  namelinJcoord = Symbol("jac", field, "_lin_coord!")
  namenlnJstruct = Symbol("jac", field, "_nln_structure!")
  namenlnJcoord = Symbol("jac", field, "_nln_coord!")

  nameJ = Symbol("jac", field)

  nccnnzj = Symbol("nnzj", field)
  nccnlinnnzj = Symbol(field, "lin_nnzj")
  nccnnlnnnzj = Symbol(field, "nln_nnzj")

  nameJprod = Symbol("j", field, "prod")
  nameJprodin = Symbol("j", field, "prod!")
  nameJprodlin = Symbol("j", field, "prod_lin!")
  nameJprodnln = Symbol("j", field, "prod_nln!")

  nameJtprod = Symbol("j", field, "tprod")
  nameJtprodin = Symbol("j", field, "tprod!")
  nameJtprodlin = Symbol("j", field, "tprod_lin")
  nameJtprodnln = Symbol("j", field, "tprod_nln")
  nameJtprodlinin = Symbol("j", field, "tprod_lin!")
  nameJtprodnlnin = Symbol("j", field, "tprod_nln!")

  nameJop = Symbol("jac", field, "_op")
  nameJopin = Symbol("j", field, "_op!")

  nameH = Symbol("hess", field)
  nameHop = Symbol("hess", field, "_op")
  nameHopin = Symbol("hess", field, "_op!")
  nameHprod = Symbol("h", field, "prod")
  nameHprodin = Symbol("h", field, "prod!")
  nameHstruct = Symbol("hess", field, "_structure")
  nameHstructin = Symbol("hess", field, "_structure!")
  nameHcoord = Symbol("hess", field, "_coord")
  nameHcoordin = Symbol("hess", field, "_coord!")

  ncccount = Symbol("neval_cons", field)
  nccJcount = Symbol("neval_jac", field)
  nccJprodcount = Symbol("neval_j", field, "prod")
  nccJtprodcount = Symbol("neval_j", field, "tprod")

  @eval begin
    @doc """
        c = $($name)(nlp, x, c)

    Evaluate the constraints of $($name) at `x`.
    """
    function $name(nlp::AbstractMPCCModel, x::AbstractVector)
      c = similar(x, nlp.cc_meta.ncc)
      return $namein(nlp, x, c)
    end
    @doc """
        c = $($name)(nlp, x, c)

    Evaluate the constraints of $($name) at `x` in place.
    """
    function $namein(nlp::AbstractMPCCModel, x::AbstractVector, cx::AbstractVector)
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.ncc cx
      increment!(nlp, $ncccount)
      nlp.meta.$nccnlin > 0 && $namelin(nlp, x, view(cx, nlp.meta.$ncclin))
      nlp.meta.$nccnnln > 0 && $namenln(nlp, x, view(cx, nlp.meta.$nccnln))
      return cx
    end

    """
        (rows,cols) = $($nameJstruct)(nlp)

    Return the structure of the constraints Jacobian in sparse coordinate format.
    """
    function $nameJstruct(nlp::AbstractMPCCModel)
      rows = Vector{Int}(undef, nlp.cc_meta.$nccnnzj)
      cols = Vector{Int}(undef, nlp.cc_meta.$nccnnzj)
      $nameJstructin(nlp, rows, cols)
    end

    """
        $($nameJstructin)(nlp, rows, cols)

    Return the structure of the constraints Jacobian in sparse coordinate format in place.
    """
    function $nameJstructin(
      nlp::AbstractMPCCModel,
      rows::AbstractVector{T},
      cols::AbstractVector{T},
    ) where {T}
      @lencheck nlp.cc_meta.$nccnnzj rows cols
      lin_ind = 1:(nlp.cc_meta.$nccnlinnnzj)
      nlp.cc_meta.$nccnlin > 0 &&
        $namelinJstruct(nlp, view(rows, lin_ind), view(cols, lin_ind))
      for i in lin_ind
        rows[i] += count(x < nlp.cc_meta.$nccnlin[rows[i]] for x in nlp.cc_meta.$nccnnln)
      end
      if nlp.cc_meta.$nccnnln > 0
        nln_ind =
          (nlp.cc_meta.$nccnlinnnzj+1):(nlp.cc_meta.$nccnlinnnzj+nlp.cc_meta.$nccnnlnnnzj)
        $namenlnJstruct(nlp, view(rows, nln_ind), view(cols, nln_ind))
        for i in nln_ind
          rows[i] += count(x < nlp.cc_meta.$nccnnln[rows[i]] for x in nlp.cc_meta.$nccnlin)
        end
      end
      return rows, cols
    end

    """
        vals = $($nameJcoordin)(nlp, x, vals)

    Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format,
    rewriting `vals`.
    """
    function $nameJcoordin(nlp::AbstractMPCCModel, x::AbstractVector, vals::AbstractVector)
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.$nccnnzj vals
      increment!(nlp, $nccJcount)
      lin_ind = 1:(nlp.cc_meta.$nccnlinnnzj)
      nlp.cc_meta.$nccnlin > 0 && $namelinJcoord(nlp, x, view(vals, lin_ind))
      nln_ind =
        (nlp.cc_meta.$nccnlinnnzj+1):(nlp.cc_meta.$nccnlinnnzj+nlp.cc_meta.$nccnnlnnnzj)
      nlp.cc_meta.$nccnnln > 0 && $namenlnJcoord(nlp, x, view(vals, nln_ind))
      return vals
    end

    """
        vals = $($nameJcoord)(nlp, x)

    Evaluate ``J(x)``, the constraints Jacobian at `x` in sparse coordinate format.
    """
    function $nameJcoord(nlp::AbstractMPCCModel{T,S}, x::AbstractVector) where {T,S}
      @lencheck nlp.meta.nvar x
      vals = S(undef, nlp.cc_meta.$nccnnzj)
      return $nameJcoordin(nlp, x, vals)
    end

    """
        Jx = $($nameJ)(nlp, x)

    Evaluate ``J(x)``, the constraints Jacobian at `x` as a sparse matrix.
    """
    function $nameJ(nlp::AbstractMPCCModel, x::AbstractVector)
      @lencheck nlp.meta.nvar x
      rows, cols = $nameJstruct(nlp)
      vals = $nameJcoord(nlp, x)
      sparse(rows, cols, vals, nlp.cc_meta.ncc, nlp.meta.nvar)
    end

    """
        Jv = $($nameJprod)(nlp, x, v)

    Evaluate ``J(x)v``, the Jacobian-vector product at `x`.
    """
    function $nameJprod(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      v::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x v
      Jv = S(undef, nlp.cc_meta.ncc)
      return $nameJprodin(nlp, x, v, Jv)
    end

    """
        Jv = $($nameJprodin)(nlp, x, v, Jv)

    Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.
    """
    function $nameJprodin(
      nlp::AbstractMPCCModel,
      x::AbstractVector,
      v::AbstractVector,
      Jv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x v
      @lencheck nlp.cc_meta.ncc Jv
      increment!(nlp, $nccJprodcount)
      nlp.cc_meta.$nccnlin > 0 && $nameJprodlin(nlp, x, v, view(Jv, nlp.cc_meta.$ncclin))
      nlp.cc_meta.$nccnnln > 0 && $nameJprodnln(nlp, x, v, view(Jv, nlp.cc_meta.$nccnln))
      return Jv
    end

    """
        Jtv = $($nameJtprod)(nlp, x, v)

    Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x`.
    """
    function $nameJtprod(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      v::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.ncc v
      Jtv = S(undef, nlp.meta.nvar)
      return $nameJtprodin(nlp, x, v, Jtv)
    end

    """
        Jtv = $($nameJtprodin)(nlp, x, v, Jtv)

    Evaluate ``J(x)^Tv``, the transposed-Jacobian-vector product at `x` in place.
    If the problem has linear and nonlinear constraints, this function allocates.
    """
    function $nameJtprodin(
      nlp::AbstractMPCCModel,
      x::AbstractVector,
      v::AbstractVector,
      Jtv::AbstractVector,
    )
      @lencheck nlp.meta.nvar x Jtv
      @lencheck nlp.cc_meta.ncc v
      increment!(nlp, $nccJtprodcount)
      if nlp.cc_meta.$nccnnln == 0
        $nameJtprodlinin(nlp, x, v, Jtv)
      elseif nlp.cc_meta.$nccnlin == 0
        $nameJtprodnlnin(nlp, x, v, Jtv)
      elseif nlp.cc_meta.$nccnlin >= nlp.cc_meta.$nccnnln
        $nameJtprodlinin(nlp, x, view(v, nlp.cc_meta.$ncclin), Jtv)
        if nlp.meta.nnln > 0
          Jtv .+= $nameJtprodnln(nlp, x, view(v, nlp.cc_meta.$nccnln))
        end
      else
        $nameJtprodnlnin(nlp, x, view(v, nlp.cc_meta.$nccnln), Jtv)
        if nlp.cc_meta.$nccnlin > 0
          Jtv .+= $nameJtprodlin(nlp, x, view(v, nlp.cc_meta.$ncclin))
        end
      end
      return Jtv
    end

    """
        J = $($nameJop)(nlp, x)

    Return the Jacobian at `x` as a linear operator.
    The resulting object may be used as if it were a matrix, e.g., `J * v` or
    `J' * v`.
    """
    function $nameJop(nlp::AbstractMPCCModel{T,S}, x::AbstractVector) where {T,S}
      @lencheck nlp.meta.nvar x
      Jv = S(undef, nlp.cc_meta.ncc)
      Jtv = S(undef, nlp.meta.nvar)
      return $nameJopin(nlp, x, Jv, Jtv)
    end

    """
        J = $($nameJopin)(nlp, x, Jv, Jtv)

    Return the Jacobian at `x` as a linear operator.
    The resulting object may be used as if it were a matrix, e.g., `J * v` or
    `J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
    operations.
    """
    function $nameJopin(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector{T},
      Jv::AbstractVector,
      Jtv::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x Jtv
      @lencheck nlp.cc_meta.ncc Jv
      prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
        $nameJprodin(nlp, x, v, Jv)
        if β == 0
          @. res = α * Jv
        else
          @. res = α * Jv + β * res
        end
        return res
      end
      ctprod! = @closure (res, v, α, β) -> begin
        $nameJtprodin(nlp, x, v, Jtv)
        if β == 0
          @. res = α * Jtv
        else
          @. res = α * Jtv + β * res
        end
        return res
      end
      return LinearOperator{T}(
        nlp.cc_meta.ncc,
        nlp.meta.nvar,
        false,
        false,
        prod!,
        ctprod!,
        ctprod!,
      )
    end

    """
        (rows,cols) = $($nameHstruct)(nlp)

    Return the structure of the Lagrangian Hessian in sparse coordinate format.
    """
    function $nameHstruct(nlp::AbstractMPCCModel)
      rows = Vector{Int}(undef, nlp.meta.nnzh)
      cols = Vector{Int}(undef, nlp.meta.nnzh)
      $nameHstructin(nlp, rows, cols)
    end

    """
        $($nameHstructin)(nlp, rows, cols)

    Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
    """
    function $nameHstructin end

    """
        vals = $($nameHcoordin)(nlp, x, y, vals)

    Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format, overwriting `vals`.
    Only the lower triangle is returned.
    """
    function $nameHcoordin end

    """
        vals = $($nameHcoord)(nlp, x, y)

    Evaluate the Lagrangian Hessian at `(x,y)` in sparse coordinate format.
    Only the lower triangle is returned.
    """
    function $nameHcoord(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      y::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.ncc y
      vals = S(undef, nlp.meta.nnzh)
      return $nameHcoordin(nlp, x, y, vals)
    end

    """
        Hx = $($nameH)(nlp, x, y)

    Evaluate the Lagrangian Hessian at `(x,y)` as a sparse matrix.
    A `Symmetric` object wrapping the lower triangle is returned.
    """
    function $nameH(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      y::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.ncc y
      rows, cols = $nameHstruct(nlp)
      vals = $nameHcoord(nlp, x, y)
      Symmetric(sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar), :L)
    end

    """
        Hv = $($nameHprod)(nlp, x, y, v)

    Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v`.
    """
    function $nameHprod(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      y::AbstractVector,
      v::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x v
      @lencheck nlp.cc_meta.ncc y
      Hv = S(undef, nlp.meta.nvar)
      return $nameHprodin(nlp, x, y, v, Hv)
    end

    """
        Hv = $($nameHprodin)(nlp, x, y, v, Hv)

    Evaluate the product of the Lagrangian Hessian at `(x,y)` with the vector `v` in
    place.
    """
    function $nameHprodin end

    """
        H = $($nameHop)(nlp, x, y)

    Return the Lagrangian Hessian at `(x,y)` as a linear operator. The resulting object may be used as if it were a
    matrix, e.g., `H * v`.
    """
    function $nameHop(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      y::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x
      @lencheck nlp.cc_meta.ncc y
      Hv = S(undef, nlp.meta.nvar)
      return $nameHopin(nlp, x, y, Hv)
    end

    """
        H = $($nameHopin)(nlp, x, y, Hv)

    Return the Lagrangian Hessian at `(x,y)` with objective function scaled by
    `obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
    object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
    used as preallocated storage for the operation.
    """
    function $nameHopin(
      nlp::AbstractMPCCModel{T,S},
      x::AbstractVector,
      y::AbstractVector,
      Hv::AbstractVector,
    ) where {T,S}
      @lencheck nlp.meta.nvar x Hv
      @lencheck nlp.cc_meta.ncc y
      prod! = @closure (res, v, α, β) -> begin
        $nameHprodin(nlp, x, y, v, Hv)
        if β == 0
          @. res = α * Hv
        else
          @. res = α * Hv + β * res
        end
        return res
      end
      return LinearOperator{T}(
        nlp.meta.nvar,
        nlp.meta.nvar,
        true,
        true,
        prod!,
        prod!,
        prod!,
      )
    end

  end
end

function NLPModels.hess_coord(
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

function NLPModels.hess(
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

function NLPModels.hprod(
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

function NLPModels.hess_op(
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

function NLPModels.hess_op!(
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
