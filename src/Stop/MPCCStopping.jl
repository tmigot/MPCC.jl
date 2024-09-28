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
mutable struct MPCCStopping{Pb,M,SRC,T,MStp,LoS} <:
               Stopping.AbstractStopping{Pb,M,SRC,T,MStp,LoS}

  # problem
  pb::Pb

  # Common parameters
  meta::M
  stop_remote::SRC

  # current state of the problem
  current_state::T

  # Stopping of the main problem, or nothing
  main_stp::MStp

  # History of states
  listofstates::LoS

  # User-specific structure
  stopping_user_struct::AbstractDict
end

function MPCCStopping(
  pb::AbstractMPCCModel,
  current_state::Stopping.AbstractState;
  stop_remote::Stopping.AbstractStopRemoteControl = Stopping.StopRemoteControl(),
  main_stp::Stopping.AbstractStopping = Stopping.VoidStopping(),
  list::Stopping.AbstractListofStates = Stopping.VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  kwargs...,
)

  if !(isempty(kwargs))
    meta = Stopping.StoppingMeta(;
      max_cntrs = _init_max_counters_mpcc(),
      optimality_check = MStat,
      kwargs...,
    )
  end

  return MPCCStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

get_pb(stp::MPCCStopping) = stp.pb
get_meta(stp::MPCCStopping) = stp.meta
get_remote(stp::MPCCStopping) = stp.stop_remote
get_state(stp::MPCCStopping) = stp.current_state
get_main_stp(stp::MPCCStopping) = stp.main_stp
get_list_of_states(stp::MPCCStopping) = stp.listofstates
get_user_struct(stp::MPCCStopping) = stp.stopping_user_struct

"""
MPCCStopping(pb): additional default constructor
The function creates a Stopping where the State is by default and the
optimality function is the function KKT().

key arguments are forwarded to the classical constructor.
"""
function MPCCStopping(pb::AbstractMPCCModel; n_listofstates::Integer = 0, kwargs...)

  #Create a default MPCCAtX
  nlp_at_x = MPCCAtX(pb.meta.x0, pb.meta.y0)
  admissible = (x, y; kwargs...) -> MStat(x, y; kwargs...)

  if n_listofstates > 0 && :list ∉ keys(kwargs)
    list = Stopping.ListofStates(n_listofstates, Val{typeof(nlp_at_x)}())
    return Stopping.NLPStopping(
      pb,
      nlp_at_x,
      list = list,
      optimality_check = KKT;
      kwargs...,
    )
  end

  return MPCCStopping(pb, nlp_at_x, optimality_check = admissible; kwargs...)
end

"""
_init_max_counters_mpcc(): initialize the maximum number of evaluations on each of
                        the functions present in the MPCCCounters.
"""
function _init_max_counters_mpcc(;
  obj::Int64 = 40000,
  grad::Int64 = 40000,
  cons::Int64 = 40000,
  consG::Int64 = 40000,
  consH::Int64 = 40000,
  jcon::Int64 = 40000,
  jgrad::Int64 = 40000,
  jac::Int64 = 40000,
  jacG::Int64 = 40000,
  jacH::Int64 = 40000,
  jprod::Int64 = 40000,
  jGprod::Int64 = 40000,
  jHprod::Int64 = 40000,
  jtprod::Int64 = 40000,
  jGtprod::Int64 = 40000,
  jHtprod::Int64 = 40000,
  hess::Int64 = 40000,
  hessG::Int64 = 40000,
  hessH::Int64 = 40000,
  hprod::Int64 = 40000,
  hGprod::Int64 = 40000,
  hHprod::Int64 = 40000,
  jhprod::Int64 = 40000,
  sum::Int64 = 40000 * 11,
)

  cntrs = Dict([
    (:neval_obj, obj),
    (:neval_grad, grad),
    (:neval_cons, cons),
    (:neval_jcon, jcon),
    (:neval_consG, consG),
    (:neval_consH, consH),
    (:neval_jgrad, jgrad),
    (:neval_jac, jac),
    (:neval_jacG, jacG),
    (:neval_jacH, jacH),
    (:neval_jprod, jprod),
    (:neval_jtprod, jtprod),
    (:neval_jGprod, jGprod),
    (:neval_jGtprod, jGtprod),
    (:neval_jHprod, jHprod),
    (:neval_jHtprod, jHtprod),
    (:neval_hess, hess),
    (:neval_hprod, hprod),
    (:neval_hessG, hessG),
    (:neval_hessH, hessH),
    (:neval_hGprod, hGprod),
    (:neval_hHprod, hHprod),
    (:neval_jhprod, jhprod),
    (:neval_sum, sum),
  ])

  return cntrs
end

function Stopping.fill_in!(
  stp::MPCCStopping,
  x::T;
  fx::Union{eltype(T),Nothing} = nothing,
  gx::Union{T,Nothing} = nothing,
  Hx = nothing,
  cx::Union{T,Nothing} = nothing,
  Jx = nothing,
  lambda::Union{T,Nothing} = nothing,
  mu::Union{T,Nothing} = nothing,
  lambdaG::Union{T,Nothing} = nothing,
  lambdaH::Union{T,Nothing} = nothing,
  cGx::Union{T,Nothing} = nothing,
  JGx = nothing,
  cHx::Union{T,Nothing} = nothing,
  JHx = nothing,
  matrix_info::Bool = true,
  convert::Bool = true,
  kwargs...,
) where {T<:AbstractVector}
  gfx = isnothing(fx) ? obj(stp.pb, x) : fx
  ggx = isnothing(gx) ? grad(stp.pb, x) : gx

  if isnothing(Hx) && matrix_info
    gHx = hess(stp.pb, x).data
  else
    gHx = isnothing(Hx) ? zeros(eltype(T), 0, 0) : Hx
  end

  if stp.pb.meta.ncon > 0
    gJx = if !isnothing(Jx)
      Jx
    elseif typeof(stp.current_state.Jx) <: LinearOperator
      jac_op(stp.pb, x)
    else # typeof(stp.current_state.Jx) <: SparseArrays.SparseMatrixCSC
      jac(stp.pb, x)
    end
    gcx = isnothing(cx) ? cons(stp.pb, x) : cx
  else
    gJx = stp.current_state.Jx
    gcx = stp.current_state.cx
  end

  #update the Lagrange multiplier if one of the 2 is asked
  if (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) && (isnothing(lambda) || isnothing(mu))
    lb, lc = Stopping._compute_mutliplier(stp.pb, x, ggx, gcx, gJx; kwargs...)
  else
    lb = if isnothing(mu) & has_bounds(stp.pb)
      zeros(eltype(T), get_nvar(stp.pb))
    elseif isnothing(mu) & !has_bounds(stp.pb)
      zeros(eltype(T), 0)
    else
      mu
    end
    lc = isnothing(lambda) ? zeros(eltype(T), get_ncon(stp.pb)) : lambda
  end

  gcGx, gcHx, gJGx, gJHx = cGx, cHx, JGx, JHx
  if stp.pb.cc_meta.ncc > 0
    gcGx = isnothing(cGx) ? consG(stp.pb, x) : cGx
    gcHx = isnothing(cHx) ? consH(stp.pb, x) : cHx
    gJGx = isnothing(JGx) ? jacG(stp.pb, x) : JGx
    gJHx = isnothing(JHx) ? jacH(stp.pb, x) : JHx
  end

  #update the Lagrange multiplier if one of the 2 is asked
  #if (stp.pb.cc_meta.ncc > 0 && stp.pb.meta.ncon > 0 && has_bounds(stp.pb)) && (lambdaG == nothing || lambdaH == nothing ||lambda == nothing || mu == nothing)
  if (stp.pb.cc_meta.ncc > 0) &&
     (lambdaG == nothing || lambdaH == nothing || lambda == nothing || mu == nothing)
    lb, lc, lG, lH = Stopping._compute_mutliplier(
      stp.pb,
      x,
      ggx,
      gcx,
      gJx,
      gcGx,
      gJGx,
      gcHx,
      gJHx,
      kwargs...,
    )
    #elseif (stp.pb.meta.ncon > 0 && has_bounds(stp.pb)) && (lambda == nothing || mu == nothing)
  elseif (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) &&
         (lambda == nothing || mu == nothing)
    lb, lc = Stopping._compute_mutliplier(stp.pb.mp, x, ggx, gcx, gJx, kwargs...)
    lG, lH = lambdaG, lambdaH
  elseif stp.pb.meta.ncon == 0 && !has_bounds(stp.pb) && lambda == nothing
    lb, lc = mu, stp.current_state.lambda
    lG, lH = lambdaG, lambdaH
  else
    lb, lc = mu, lambda
    lG, lH = lambdaG, lambdaH
  end

  return Stopping.update!(
    stp.current_state,
    x = x,
    fx = gfx,
    gx = ggx,
    Hx = gHx,
    cx = gcx,
    Jx = gJx,
    mu = lb,
    lambda = lc,
    lambdaG = lG,
    lambdaH = lH,
    cGx = gcGx,
    cHx = gcHx,
    JGx = gJGx,
    JHx = gJHx,
  )
end

"""
_resources_check!: check if the optimization algorithm has exhausted the resources.
                   This is the MPCC specialized version that takes into account
                   the evaluation of the functions following the sum_counters
                   structure from MPCC.

Note: function uses counters in stp.pb, and update the counters in the state
"""
function Stopping._resources_check!(
  stp::MPCCStopping{Pb,M,SRC,T,MStp,LoS},
  x::S,
) where {Pb<:AbstractMPCCModel,M,SRC,T,MStp,LoS,S}
  max_cntrs = stp.meta.max_cntrs

  if length(max_cntrs) == 0
    return stp.meta.resources
  end

  # check all the entries in the counter
  max_f = Stopping.check_entries_counters(stp.pb, max_cntrs)

  # Maximum number of function and derivative(s) computation
  if :neval_sum in keys(max_cntrs)
    max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]
  end

  # global user limit diagnostic
  if (max_evals || max_f)
    stp.meta.resources = true
  end

  return stp.meta.resources
end

function Stopping.check_entries_counters(nlp::AbstractMPCCModel, max_cntrs)
  for f in keys(max_cntrs)
    if f in fieldnames(Counters)
      if eval(f)(nlp)::Int > max_cntrs[f]
        return true
      end
    end
    if f in fieldnames(MPCCCounters)
      if eval(f)(nlp)::Int > max_cntrs[f]
        return true
      end
    end
  end
  return false
end

function Stopping._unbounded_problem_check!(
  stp::MPCCStopping{Pb,M,SRC,MPCCAtX{Score,S,T},MStp,LoS},
  x::AbstractVector,
) where {Pb,M,SRC,MStp,LoS,Score,S,T}
  if isnan(get_fx(stp.current_state))
    stp.current_state.fx = obj(stp.pb, x)
  end

  if stp.pb.meta.minimize
    f_too_large = get_fx(stp.current_state) <= -stp.meta.unbounded_threshold
  else
    f_too_large = get_fx(stp.current_state) >= stp.meta.unbounded_threshold
  end

  if f_too_large
    stp.meta.unbounded_pb = true
  end

  return stp.meta.unbounded_pb
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
