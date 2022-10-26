"""
Type: MPCCAtX
Methods: update!, reinit!

MPCCAtX contains the information concerning a nonlinear problem at
the iteration x.
min_{x ∈ ℜⁿ} f(x)
subject to lcon <= c(x) <= ucon, lvar <= x <= uvar, lG <= G(x) _|_ H(x) >= lH.

Basic information is:
 - x : the current candidate for solution to our original problem
 - fx : which is the funciton evaluation at x
 - gx : which is the gradient evaluation at x
 - Hx : which is the hessian representation at x

 - mu : Lagrange multiplier of the bounds constraints

 - cx : evaluation of the constraint function at x
 - Jx : jacobian matrix of the constraint function at x
 - lambda : Lagrange multiplier of the constraints

 - current_time : time
 - evals : number of evaluations of the function

Note: * by default, unknown entries are set to nothing (except evals).
      * All these information (except for x and lambda) are optionnal and need to be update when
        required. The update is done trhough the update! function.
      * x and lambda are mandatory entries. If no constraints lambda = [].
      * The constructor check the size of the entries.
"""
mutable struct MPCCAtX{Score, S, T <: AbstractVector} <: AbstractState{S, T}

    #Unconstrained State
    x::T     # current point
    fx::S   # objective function
    gx::T    # gradient size: x
    Hx # hessian size: |x| x |x|

    #Bounds State
    mu::T    # Lagrange multipliers with bounds size of |x|

    #Constrained State
    cx::T    # vector of constraints lc <= c(x) <= uc
    Jx  # jacobian matrix, size: |lambda| x |x|
    lambda::T    # Lagrange multipliers

    #Complementarity State
    cGx::T    # vector of constraints lG <= G(x) <= uG
    JGx  # jacobian matrix, size: |ncc| x |x|
    lambdaG::T   # Lagrange multipliers
    cHx::T    # vector of constraints lH <= H(x) <= uH
    JHx  # jacobian matrix, size: |ncc| x |x|
    lambdaH::T   # Lagrange multipliers

    d::T#search direction
    res::T#residual

    #Resources State
    current_time::Float64
    current_score::Score

    function MPCCAtX(
        x::T,
        lambda::T,
        current_score::Score = _init_field(eltype(T));
        fx::eltype(T) = _init_field(eltype(T)),
        gx::T = _init_field(T),
        Hx = _init_field(Matrix{eltype(T)}),
        mu::T = _init_field(T),
        cx::T = _init_field(T),
        Jx = _init_field(SparseMatrixCSC{eltype(T), Int64}),
        lambdaG::T = _init_field(T),
        lambdaH::T = _init_field(T),
        cGx::T = _init_field(T),
        JGx = _init_field(SparseMatrixCSC{eltype(T), Int64}),
        cHx::T = _init_field(T),
        JHx = _init_field(SparseMatrixCSC{eltype(T), Int64}),
        d::T = _init_field(T),
        res::T = _init_field(T),
        current_time::Float64 = NaN,
    ) where {Score, T <: AbstractVector}

        return new{Score, eltype(T), T}(
            x,
            fx,
            gx,
            Hx,
            mu,
            cx,
            Jx,
            lambda,
            cGx,
            JGx,
            lambdaG,
            cHx,
            JHx,
            lambdaH,
            d,
            res,
            current_time,
            current_score,
        )
    end
end

function reinit!(stateatx::MPCCAtX, x::Vector, l::Vector; kwargs...)

    for k ∈ fieldnames(MPCCAtX)
        if !(k ∈ [:x, :lambda])
            setfield!(stateatx, k, _init_field(typeof(getfield(stateatx, k))))
        end
    end

    setfield!(stateatx, :x, x)
    setfield!(stateatx, :lambda, l)
  
    if length(kwargs) == 0
      return stateatx #save the update! call if no other kwargs than x
    end
  
    return update!(stateatx; kwargs...)
end

function reinit!(stateatx::MPCCAtX; kwargs...)
    return reinit!(stateatx, stateatx.x, stateatx.lambda; kwargs...)
end
