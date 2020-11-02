using Stopping

import Stopping: _init_max_counters, fill_in!, _resources_check!, _unbounded_problem_check!, _optimality_check
import Stopping: start!, stop!, update_and_start!, update_and_stop!, reinit!, status

"""
Type: MPCCStopping (specialization of GenericStopping)
Methods: start!, stop!, update_and_start!, update_and_stop!, fill_in!, reinit!, status

Stopping structure for non-linear programming problems using NLPModels.
    Input :
       - pb         : an AbstractMPCCModel
       - state      : The information relative to the problem, see GenericState
       - (opt) meta : Metadata relative to stopping criterion, see *StoppingMeta*.
       - (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.
       - (opt) listofstates : ListStates designed to store the history of States.
       - (opt) user_specific_struct : Contains any structure designed by the user.

`MPCCStopping(:: AbstractMPCCModel, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta(), main_stp :: Union{AbstractStopping, Nothing} = nothing, list :: Union{ListStates, Nothing} = nothing, user_specific_struct :: Any = nothing, kwargs...)`


 Note:
 * optimality_check : takes two inputs (AbstractMPCCModel, MPCCAtX)
 and returns a Float64 to be compared at 0.
 * designed for MPCCAtX State. Constructor checks that the State has the
 required entries.

 Warning:
 * optimality_check does not necessarily fill in the State.
 """
mutable struct MPCCStopping <: AbstractStopping

    # problem
    pb :: AbstractMPCCModel

    # Common parameters
    meta      :: AbstractStoppingMeta

    # current state of the problem
    current_state :: AbstractState

    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    listofstates :: Union{ListStates, Nothing}

    # User-specific structure
    user_specific_struct :: Any

    function MPCCStopping(pb            :: AbstractMPCCModel,
                         current_state  :: AbstractState;
                         meta           :: AbstractStoppingMeta = StoppingMeta(;max_cntrs = _init_max_counters_mpcc(), optimality_check = MStat),
                         main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                         listofstates   :: Union{ListStates, Nothing} = nothing,
                         user_specific_struct :: Any = nothing,
                         kwargs...)

        if !(isempty(kwargs))
           meta = StoppingMeta(;max_cntrs = _init_max_counters_mpcc(), optimality_check = MStat, kwargs...)
        end

        #current_state is an AbstractState with requirements
        try
            current_state.evals
            current_state.fx, current_state.gx, current_state.Hx
            #if there are bounds:
            current_state.mu
            if pb.meta.ncon > 0 #if there are constraints
               current_state.Jx, current_state.cx, current_state.lambda
            end
        catch
            throw("error: missing entries in the given current_state")
        end

        return new(pb, meta, current_state, main_stp, listofstates, user_specific_struct)
    end

end

"""
MPCCStopping(pb): additional default constructor
The function creates a Stopping where the State is by default and the
optimality function is the function KKT().

key arguments are forwarded to the classical constructor.
"""
function MPCCStopping(pb :: AbstractMPCCModel; kwargs...)
    
 #Create a default MPCCAtX
 nlp_at_x = MPCCAtX(pb.meta.x0, pb.meta.y0)
 admissible = (x,y; kwargs...) -> MStat(x,y; kwargs...)

 return MPCCStopping(pb, nlp_at_x, optimality_check = admissible; kwargs...)
end

"""
_init_max_counters_mpcc(): initialize the maximum number of evaluations on each of
                        the functions present in the MPCCCounters.
"""
function _init_max_counters_mpcc(; obj     :: Int64 = 40000,
                                   grad    :: Int64 = 40000,
                                   cons    :: Int64 = 40000,
                                   consG   :: Int64 = 40000,
                                   consH   :: Int64 = 40000,
                                   jcon    :: Int64 = 40000,
                                   jgrad   :: Int64 = 40000,
                                   jac     :: Int64 = 40000,
                                   jacG    :: Int64 = 40000,
                                   jacH    :: Int64 = 40000,
                                   jprod   :: Int64 = 40000,
                                   jGprod  :: Int64 = 40000,
                                   jHprod  :: Int64 = 40000,
                                   jtprod  :: Int64 = 40000,
                                   jGtprod :: Int64 = 40000,
                                   jHtprod :: Int64 = 40000,
                                   hess    :: Int64 = 40000,
                                   hessG   :: Int64 = 40000,
                                   hessH   :: Int64 = 40000,
                                   hprod   :: Int64 = 40000,
                                   hGprod  :: Int64 = 40000,
                                   hHprod  :: Int64 = 40000,
                                   jhprod  :: Int64 = 40000,
                                   sum     :: Int64 = 40000*11)

  cntrs = Dict([(:neval_obj,       obj), (:neval_grad,   grad),
                (:neval_cons,     cons), (:neval_jcon,   jcon),
                (:neval_consG,   consG), (:neval_consH,   consH),
                (:neval_jgrad,   jgrad), (:neval_jac,    jac),
                (:neval_jacG,     jacG), (:neval_jacH,     jacH),
                (:neval_jprod,   jprod), (:neval_jtprod, jtprod),
                (:neval_jGprod, jGprod), (:neval_jGtprod, jGtprod),
                (:neval_jHprod, jHprod), (:neval_jHtprod, jHtprod),
                (:neval_hess,     hess), (:neval_hprod,  hprod),
                (:neval_hessG,   hessG), (:neval_hessH,   hessH),
                (:neval_hGprod,  hGprod),(:neval_hHprod,  hHprod),
                (:neval_jhprod, jhprod), (:neval_sum,    sum)])

 return cntrs
end

"""
fill_in!: a function that fill in the required values in the State

kwargs are forwarded to _compute_mutliplier function
"""
function fill_in!(stp     :: MPCCStopping,
                  x       :: Iterate;
                  fx      :: Iterate    = nothing,
                  gx      :: Iterate    = nothing,
                  Hx      :: Iterate    = nothing,
                  cx      :: Iterate    = nothing,
                  Jx      :: Iterate    = nothing,
                  lambda  :: Iterate    = nothing,
                  mu      :: Iterate    = nothing,
                  lambdaG :: Iterate    = nothing,
                  lambdaH :: Iterate    = nothing,
                  cGx     :: Iterate    = nothing,
                  JGx     :: MatrixType = nothing,
                  cHx     :: Iterate    = nothing,
                  JHx     :: MatrixType = nothing,
                  matrix_info :: Bool   = true,
                  kwargs...)

 gfx = fx == nothing  ? obj(stp.pb, x)   : fx
 ggx = gx == nothing  ? grad(stp.pb, x)  : gx

 if Hx == nothing && matrix_info
   gHx = hess(stp.pb, x)
 else
   gHx = Hx
 end

 if stp.pb.meta.ncon > 0
     gJx = Jx == nothing ? jac_nl(stp.pb, x)  : Jx
     gcx = cx == nothing ? cons_nl(stp.pb, x) : cx
 else
     gJx = Jx
     gcx = cx
 end

 gcGx, gcHx, gJGx, gJHx = cGx, cHx, JGx, JHx
 if stp.pb.meta.ncc > 0
     gcGx = cGx == nothing ? consG(stp.pb, x) : cGx
     gcHx = cHx == nothing ? consH(stp.pb, x) : cHx
     gJGx = JGx == nothing ? jacG(stp.pb, x)  : JGx
     gJHx = JHx == nothing ? jacH(stp.pb, x)  : JHx
 end

 #update the Lagrange multiplier if one of the 2 is asked
 #if (stp.pb.meta.ncc > 0 && stp.pb.meta.ncon > 0 && has_bounds(stp.pb)) && (lambdaG == nothing || lambdaH == nothing ||lambda == nothing || mu == nothing)
 if (stp.pb.meta.ncc > 0) && (lambdaG == nothing || lambdaH == nothing ||lambda == nothing || mu == nothing)
  lb, lc, lG, lH = _compute_mutliplier(stp.pb, x, ggx, gcx, gJx, gcGx, gJGx, gcHx, gJHx, kwargs...)
 #elseif (stp.pb.meta.ncon > 0 && has_bounds(stp.pb)) && (lambda == nothing || mu == nothing)
 elseif (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) && (lambda == nothing || mu == nothing)
  lb, lc = _compute_mutliplier(stp.pb.mp, x, ggx, gcx, gJx, kwargs...)
  lG, lH = lambdaG, lambdaH
 elseif  stp.pb.meta.ncon == 0 && !has_bounds(stp.pb) && lambda == nothing
  lb, lc = mu, stp.current_state.lambda
  lG, lH = lambdaG, lambdaH
 else
  lb, lc = mu, lambda
  lG, lH = lambdaG, lambdaH
 end

 return update!(stp.current_state, x=x, fx = gfx,    gx = ggx, Hx = gHx,
                                        cx = gcx,    Jx = gJx, mu = lb,
                                        lambda = lc, lambdaG = lG, lambdaH = lH,
                                        cGx = gcGx, cHx = gcHx,
                                        JGx = gJGx, JHx = gJHx)
end

"""
_resources_check!: check if the optimization algorithm has exhausted the resources.
                   This is the MPCC specialized version that takes into account
                   the evaluation of the functions following the sum_counters
                   structure from MPCC.

Note: function uses counters in stp.pb, and update the counters in the state
"""
function _resources_check!(stp    :: MPCCStopping,
                           x      :: Iterate)

  cntrs = stp.pb.counters
  update!(stp.current_state, evals = cntrs)

  max_cntrs = stp.meta.max_cntrs

  # check all the entries in the counter
  max_f = false
  for f in fieldnames(MPCCCounters)
      max_f = max_f || (getfield(cntrs, f) > max_cntrs[f])
  end

 # Maximum number of function and derivative(s) computation
 max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]

 # global user limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
_unbounded_problem_check!: This is the NLP specialized version that takes into account
                   that the problem might be unbounded if the objective or the
                   constraint function are unbounded.

Note: * evaluate the objective function if state.fx is void.
      * evaluate the constraint function if state.cx is void.
"""
function _unbounded_problem_check!(stp  :: MPCCStopping,
                                   x    :: Iterate)

 if stp.current_state.fx == nothing
	 stp.current_state.fx = obj(stp.pb, x)
 end
 f_too_large = norm(stp.current_state.fx) >= stp.meta.unbounded_threshold

 c_too_large = false
 if stp.pb.meta.ncon != 0 #if the problems has constraints, check |c(x)|
  if stp.current_state.cx == nothing
   stp.current_state.cx = cons(stp.pb, x)
  end
  c_too_large = norm(stp.current_state.cx) >= abs(stp.meta.unbounded_threshold)
 end

 stp.meta.unbounded_pb = f_too_large || c_too_large

 return stp
end

################################################################################
# Nonlinear problems admissibility functions
# Available: unconstrained_check(...), optim_check_bounded(...), KKT
################################################################################
include("mpcc_admissible_functions.jl")

################################################################################
# Functions computing Lagrange multipliers of a nonlinear problem
# Available: _compute_mutliplier(...)
################################################################################
include("nlp_compute_multiplier.jl")
