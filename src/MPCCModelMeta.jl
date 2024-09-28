#####################################################################################
# MPCCModelMeta type ; based on NLPModelMeta:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl
#
#
# TODO:
# - tailor functions: print and show
# - tailor if problem is just a complementarity problem (no cost)
#####################################################################################

# Base type for metadata related to an optimization model.
abstract type AbstractMPCCModelMeta{T,S} end

"""
A composite type that represents the main features of the optimization problem

  optimize   obj(x)
  subject to lvar ≤    x    ≤ uvar
             lcon ≤ cons(x) ≤ ucon
             lcc  ≤ G(x) _|_ H(x) >= ucc

 where x        is an nvar-dimensional vector,
       obj      is the real-valued objective function,
       cons     is the vector-valued constraint function,
       optimize is either "minimize" or "maximize".

 Here, lvar, uvar, lcon and ucon are vectors. Some of their
 components may be infinite to indicate that the corresponding
 bound or general constraint is not present.
"""
struct MPCCModelMeta{T,S} <: AbstractMPCCModelMeta{T,S}

  ncc::Int               # number of complementarity constraints
  yG::S    # initial Lagrange multipliers
  yH::S    # initial Lagrange multipliers
  lccG::S    # vector of constraint lower bounds of the complementarity constraint
  lccH::S    # vector of constraint upper bounds of the complementarity constraint
  nnzjG::Int
  nnzjH::Int
end

function MPCCModelMeta{T,S}(
  nvar,
  ncc;
  yG::S = fill!(S(undef, ncc), zero(T)),
  yH::S = fill!(S(undef, ncc), zero(T)),
  lccG::S = fill!(S(undef, ncc), T(-Inf)),
  lccH::S = fill!(S(undef, ncc), T(Inf)),
  nnzjG = ncc * nvar,
  nnzjH = ncc * nvar,
) where {T,S}
  @lencheck ncc lccG lccH yG yH
  MPCCModelMeta{T,S}(ncc, yG, yH, lccG, lccH, nnzjG, nnzjH)
end

MPCCModelMeta(nvar, ncc; yG::S = zeros(ncc), kwargs...) where {S} =
  MPCCModelMeta{eltype(S),S}(nvar, ncc, yG = yG; kwargs...)

"""
    complementarity_constrained(nlp)
    complementarity_constrained(meta)

Returns whether the problem's constraints are all inequalities.
Unconstrained problems return true.
"""
complementarity_constrained(meta::MPCCModelMeta) = meta.ncc > 0
complementarity_constrained(nlp::AbstractMPCCModel) =
  complementarity_constrained(nlp.cc_meta)
