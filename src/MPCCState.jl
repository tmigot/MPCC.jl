using Stopping

import Stopping: reinit!, update!

const Iterate           = Union{Number, Vector, Nothing}
const FloatVoid         = Union{Number, Nothing}
const MatrixType        = Union{Number, AbstractArray, AbstractMatrix, Nothing}
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
mutable struct 	MPCCAtX{T <: AbstractVector} <: AbstractState

#Unconstrained State
    x            :: T     # current point
    fx           :: Union{eltype(T),Nothing}   # objective function
    gx           :: Union{T,eltype(T),Nothing}     # gradient size: x
    Hx           :: MatrixType  # hessian size: |x| x |x|

#Bounds State
    mu           :: Union{T,eltype(T),Nothing}     # Lagrange multipliers with bounds size of |x|

#Constrained State
    cx           :: Union{T,eltype(T),Nothing}     # vector of constraints lc <= c(x) <= uc
    Jx           :: MatrixType  # jacobian matrix, size: |lambda| x |x|
    lambda       :: T    # Lagrange multipliers

#Complementarity State
    cGx           :: Union{T,eltype(T),Nothing}     # vector of constraints lG <= G(x) <= uG
    JGx           :: MatrixType  # jacobian matrix, size: |ncc| x |x|
    lambdaG       :: Union{T,eltype(T),Nothing}    # Lagrange multipliers
    cHx           :: Union{T,eltype(T),Nothing}     # vector of constraints lH <= H(x) <= uH
    JHx           :: MatrixType  # jacobian matrix, size: |ncc| x |x|
    lambdaH       :: Union{T,eltype(T),Nothing}    # Lagrange multipliers

    d            :: Union{T,eltype(T),Nothing} #search direction
    res          :: Union{T,eltype(T),Nothing} #residual

 #Resources State
    current_time   :: Union{eltype(T),Nothing}
    current_score  :: Union{T,eltype(T),Nothing}
    evals          :: MPCCCounters

 function MPCCAtX(x             :: T,
                  lambda        :: T;
                  fx            :: FloatVoid    = nothing,
                  gx            :: Iterate      = nothing,
                  Hx            :: MatrixType   = nothing,
                  mu            :: Iterate      = nothing,
                  cx            :: Iterate      = nothing,
                  Jx            :: MatrixType   = nothing,
                  lambdaG       :: Iterate      = nothing,
                  lambdaH       :: Iterate      = nothing,
                  cGx           :: Iterate      = nothing,
                  JGx           :: MatrixType   = nothing,
                  cHx           :: Iterate      = nothing,
                  JHx           :: MatrixType   = nothing,
                  d             :: Iterate      = nothing,
                  res           :: Iterate      = nothing,
                  current_time  :: FloatVoid    = nothing,
                  current_score :: Iterate     = nothing,
                  evals         :: MPCCCounters = MPCCCounters()) where T <: AbstractVector

  return new{T}(x, fx, gx, Hx, mu,
                cx, Jx, lambda,
                cGx, JGx, lambdaG,
                cHx, JHx, lambdaH,
                d, res, current_time, current_score, evals)
 end
end

"""
reinit!: function that set all the entries at void except the mandatory x

Warning: if x, lambda or evals are given as a keyword argument they will be
prioritized over the existing x, lambda and the default Counters.
"""
function reinit!(stateatx :: MPCCAtX, x :: Vector, l :: Vector; kwargs...)

 for k ∈ fieldnames(MPCCAtX)
   if !(k ∈ [:x,:lambda,:evals]) setfield!(stateatx, k, nothing) end
 end

 return update!(stateatx; x=x, lambda = l, evals = MPCCCounters(), kwargs...)
end

"""
reinit!: short version of reinit! reusing the x in the state

Warning: if x, lambda or evals are given as a keyword argument they will be
prioritized over the existing x, lambda and the default Counters.
"""
function reinit!(stateatx :: MPCCAtX; kwargs...)
 return reinit!(stateatx, stateatx.x, stateatx.lambda; kwargs...)
end
