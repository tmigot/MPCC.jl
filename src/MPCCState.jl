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
mutable struct 	MPCCAtX <: AbstractState

#Unconstrained State
    x            :: Vector     # current point
    fx           :: FloatVoid   # objective function
    gx           :: Iterate     # gradient size: x
    Hx           :: MatrixType  # hessian size: |x| x |x|

#Bounds State
    mu           :: Iterate     # Lagrange multipliers with bounds size of |x|

#Constrained State
    cx           :: Iterate     # vector of constraints lc <= c(x) <= uc
    Jx           :: MatrixType  # jacobian matrix, size: |lambda| x |x|
    lambda       :: Vector    # Lagrange multipliers

#Complementarity State
    cGx           :: Iterate     # vector of constraints lG <= G(x) <= uG
    JGx           :: MatrixType  # jacobian matrix, size: |ncc| x |x|
    lambdaG       :: Iterate    # Lagrange multipliers
    cHx           :: Iterate     # vector of constraints lH <= H(x) <= uH
    JHx           :: MatrixType  # jacobian matrix, size: |ncc| x |x|
    lambdaH       :: Iterate    # Lagrange multipliers

 #Resources State
    current_time   :: FloatVoid
    evals          :: MPCCCounters

 function MPCCAtX(x            :: Vector,
                  lambda       :: Vector;
                  fx           :: FloatVoid    = nothing,
                  gx           :: Iterate      = nothing,
                  Hx           :: MatrixType   = nothing,
                  mu           :: Iterate      = nothing,
                  cx           :: Iterate      = nothing,
                  Jx           :: MatrixType   = nothing,
                  lambdaG      :: Iterate      = nothing,
                  lambdaH      :: Iterate      = nothing,
                  cGx          :: Iterate      = nothing,
                  JGx          :: MatrixType   = nothing,
                  cHx          :: Iterate      = nothing,
                  JHx          :: MatrixType   = nothing,
                  current_time :: FloatVoid    = nothing,
                  evals        :: MPCCCounters = MPCCCounters())

  return new(x, fx, gx, Hx, mu, cx, Jx, lambda, cGx, JGx, lambdaG, cHx, JHx, lambdaH, current_time, evals)
 end
end

"""
reinit!: function that set all the entries at void except the mandatory x

Warning: if x, lambda or evals are given as a keyword argument they will be
prioritized over the existing x, lambda and the default Counters.
"""
function reinit!(stateatx :: MPCCAtX, x :: Vector, l :: Vector; kwargs...)

 for k ∈ fieldnames(typeof(stateatx))
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
