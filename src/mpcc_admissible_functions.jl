"""
constrained: return the violation of the KKT conditions
length(lambda) > 0
"""
function _grad_lagrangian(pb    :: AbstractMPCCModel,
                          state :: MPCCAtX)

 if (pb.meta.ncon + pb.meta.ncc) == 0 & !has_bounds(pb)
  return state.gx
 elseif (pb.meta.ncon + pb.meta.ncc) == 0
  return state.gx + state.mu
 elseif pb.meta.ncc == 0
  return state.gx + state.mu + state.Jx' * state.lambda
 else #pb.meta.ncon + pb.meta.ncc > 0
  return state.gx + state.mu + state.Jx' * state.lambda - state.JGx' * state.lambdaG - state.JHx' * state.lambdaH
 end
end

function _sign_multipliers_bounds(pb    :: AbstractMPCCModel,
                                  state :: MPCCAtX)
 if has_bounds(pb)
  return vcat(min.(max.( state.mu,0.0), - state.x + pb.meta.uvar),
              min.(max.(-state.mu,0.0),   state.x - pb.meta.lvar))
 else
  return zeros(0)
 end
end

function _sign_multipliers_nonlin(pb    :: AbstractMPCCModel,
                                  state :: MPCCAtX)
 if pb.meta.ncon == 0
  return zeros(0)
 else
  return vcat(min.(max.(  state.lambda, 0.0), - state.cx + pb.meta.ucon),
              min.(max.(- state.lambda, 0.0),   state.cx - pb.meta.lcon))
 end
end

function _sign_multipliers_comp(pb    :: AbstractMPCCModel,
                                state :: MPCCAtX)
 if pb.meta.ncc == 0
  return zeros(0)
 else
  return vcat(min.(abs.(state.lambdaG), abs.(state.cGx - pb.meta.lccG)),
              min.(abs.(state.lambdaH), abs.(state.cHx - pb.meta.lccH)))
 end
end

function _sign_multipliers_comp_Sstat(pb    :: AbstractMPCCModel,
                                      state :: MPCCAtX,
                                      actif :: Float64)

 I = findall(x -> x<=actif, 0.5*(state.cGx + state.cHx))
 pos = max.(max.(- state.lambdaG[I], 0.0), max.(- state.lambdaH[I], 0.0))

 return pos
end

function _sign_multipliers_comp_Mstat(pb    :: AbstractMPCCModel,
                                      state :: MPCCAtX)

 eq  = max.(abs.(state.lambdaG .* state.cGx), abs.(state.lambdaH .* state.cHx))
 pos = max.(- state.lambdaG .* state.lambdaH, 0.0)
 mpo = max.(- max.(state.lambdaG, state.lambdaH), 0.0)

 return max.(eq, pos, mpo)
end

function _sign_multipliers_comp_Cstat(pb    :: AbstractMPCCModel,
                                      state :: MPCCAtX)

 eq  = max.(abs.(state.lambdaG .* state.cGx), abs.(state.lambdaH .* state.cHx))
 pos = max.(- state.lambdaG .* state.lambdaH, 0.0)

 return max.(eq, pos)
end

function _feasibility(pb    :: AbstractMPCCModel,
                      state :: MPCCAtX)
 if (pb.meta.ncon + pb.meta.ncc) == 0
  return vcat(max.(  state.x  - pb.meta.uvar, 0.0),
              max.(- state.x  + pb.meta.lvar, 0.0))
 elseif pb.meta.ncc == 0
  return vcat(max.(  state.cx  - pb.meta.ucon, 0.0),
              max.(- state.cx  + pb.meta.lcon, 0.0),
              max.(  state.x   - pb.meta.uvar, 0.0),
              max.(- state.x   + pb.meta.lvar, 0.0))
 else #pb.meta.ncon + pb.meta.ncc > 0
  return vcat(max.(  state.cx  - pb.meta.ucon, 0.0),
              max.(- state.cx  + pb.meta.lcon, 0.0),
              max.(  state.x   - pb.meta.uvar, 0.0),
              max.(- state.x   + pb.meta.lvar, 0.0),
              max.(- state.cGx + pb.meta.lccG, 0.0),
              max.(- state.cHx + pb.meta.lccH, 0.0),
              min.(  state.cGx, state.cHx))
 end
end

"""
W-stationary check:
WStat: verifies the Weak-stationary conditions
required: state.gx
+ if bounds: state.mu
+ if constraints: state.cx, state.Jx, state.lambda
+ if complementarity: state.cGx, state.JGx, state.lambdaG
                      state.cHx, state.JHx, state.lambdaH
"""
function WStat(pb    :: AbstractMPCCModel,
               state :: MPCCAtX;
               pnorm :: Float64 = Inf,
               kwargs...)

    #Check the gradient of the Lagrangian
    gLagx      = _grad_lagrangian(pb, state)
    #Check the complementarity condition for the bounds
    res_bounds = _sign_multipliers_bounds(pb, state)
    #Check the complementarity condition for the constraints
    res_nonlin = _sign_multipliers_nonlin(pb, state)
    #Check the complementarity condition for the complementarity constraints
    res_comp   = _sign_multipliers_comp(pb, state)
    #Check the feasibility
    feas       = _feasibility(pb, state)

    res = vcat(gLagx, feas, res_bounds, res_nonlin, res_comp)

    return norm(res, pnorm)
end

"""
S-stationary check:
Compute the score corresponding to:
a) W-stationarity
b) the correct sign of multipliers for indices
I_00 = { i : 0.5(G_i(x^*)+H_i(x^*)) <= actif }

required: state.gx
+ if bounds: state.mu
+ if constraints: state.cx, state.Jx, state.lambda
+ if complementarity: state.cGx, state.JGx, state.lambdaG
                      state.cHx, state.JHx, state.lambdaH
"""
function SStat(pb    :: AbstractMPCCModel,
               state :: MPCCAtX;
               pnorm :: Float64 = Inf,
               actif :: Float64 = 1e-6,
               kwargs...)

    res = WStat(pb, state, pnorm = pnorm; kwargs...)
    sign = pb.meta.ncc > 0 ? norm(_sign_multipliers_comp_Sstat(pb, state, actif), pnorm) : 0.0

    return max(res, sign)
end

"""
M-stationary check:
Compute the score corresponding to:
a) W-stationarity
b) the correct sign of multipliers for indices I_00 by checking:
         lambdaG .* lambdaH => 0
(i.e. max.(- state.lambdaG .* state.lambdaH, 0.0))
and
         max(lambdaG, lambdaH) => 0
(i.e. max.(- max.(state.lambdaG, state.lambdaH), 0.0))

required: state.gx
+ if bounds: state.mu
+ if constraints: state.cx, state.Jx, state.lambda
+ if complementarity: state.cGx, state.JGx, state.lambdaG
                      state.cHx, state.JHx, state.lambdaH
"""
function MStat(pb    :: AbstractMPCCModel,
               state :: MPCCAtX;
               pnorm :: Float64 = Inf,
               kwargs...)

    res = WStat(pb, state, pnorm = pnorm; kwargs...)
    sign = pb.meta.ncc > 0 ? norm(_sign_multipliers_comp_Mstat(pb, state), pnorm) : 0.0

    return max(res, sign)
end

"""
C-stationary check:
Compute the score corresponding to:
a) W-stationarity
b) the correct sign of multipliers for indices I_00 by checking:
         lambdaG .* lambdaH => 0
(i.e. max.(- state.lambdaG .* state.lambdaH, 0.0))

required: state.gx
+ if bounds: state.mu
+ if constraints: state.cx, state.Jx, state.lambda
+ if complementarity: state.cGx, state.JGx, state.lambdaG
                      state.cHx, state.JHx, state.lambdaH
"""
function CStat(pb    :: AbstractMPCCModel,
               state :: MPCCAtX;
               pnorm :: Float64 = Inf,
               kwargs...)

    res = WStat(pb, state, pnorm = pnorm; kwargs...)
    sign = pb.meta.ncc > 0 ? norm(_sign_multipliers_comp_Cstat(pb, state), pnorm) : 0.0

    return max(res, sign)
end
