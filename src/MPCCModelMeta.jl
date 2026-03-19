#####################################################################################
# MPCCModelMeta type ; based on NLPModelMeta:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl
#
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

  ncc::Int   # number of complementarity constraints
  Glin::Vector{Int}
  Gnlin::Int
  Gnln::Vector{Int}
  Gnnln::Int
  Hlin::Vector{Int}
  Hnlin::Int
  Hnln::Vector{Int}
  Hnnln::Int
  yG::S      # initial Lagrange multipliers
  yH::S      # initial Lagrange multipliers
  lccG::S    # vector of constraint lower bounds of the complementarity constraint
  lccH::S    # vector of constraint upper bounds of the complementarity constraint
  nnzjG::Int
  Glin_nnzj::Int
  Gnln_nnzj::Int
  nnzjH::Int
  Hlin_nnzj::Int
  Hnln_nnzj::Int
end

# Custom print and show for MPCCModelMeta
import Base: show, print

function show(io::IO, meta::MPCCModelMeta)
"""
  show(io::IO, meta::MPCCModelMeta)

Custom display for MPCCModelMeta, prints all fields in a readable format.
"""
  println(io, "MPCCModelMeta:")
  println(io, "  ncc: ", meta.ncc)
  println(io, "  Glin: ", meta.Glin)
  println(io, "  Gnlin: ", meta.Gnlin)
  println(io, "  Gnln: ", meta.Gnln)
  println(io, "  Gnnln: ", meta.Gnnln)
  println(io, "  Hlin: ", meta.Hlin)
  println(io, "  Hnlin: ", meta.Hnlin)
  println(io, "  Hnln: ", meta.Hnln)
  println(io, "  Hnnln: ", meta.Hnnln)
  println(io, "  yG: ", meta.yG)
  println(io, "  yH: ", meta.yH)
  println(io, "  lccG: ", meta.lccG)
  println(io, "  lccH: ", meta.lccH)
  println(io, "  nnzjG: ", meta.nnzjG)
  println(io, "  Glin_nnzj: ", meta.Glin_nnzj)
  println(io, "  Gnln_nnzj: ", meta.Gnln_nnzj)
  println(io, "  nnzjH: ", meta.nnzjH)
  println(io, "  Hlin_nnzj: ", meta.Hlin_nnzj)
  println(io, "  Hnln_nnzj: ", meta.Hnln_nnzj)
end

function print(io::IO, meta::MPCCModelMeta)
"""
  print(io::IO, meta::MPCCModelMeta)

Prints MPCCModelMeta using the custom show method.
"""
  show(io, meta)
end

# Utility to check if problem is complementarity-only (no cost)
function is_complementarity_only(meta::MPCCModelMeta, obj::Function)
"""
  is_complementarity_only(meta::MPCCModelMeta, obj::Function)

Returns true if the objective function is constant zero, indicating a complementarity-only problem.
Checks obj(zeros(length(meta.yG))) == 0.0 and that the result is numeric.
"""
  # If objective is constant or zero, treat as complementarity-only
  x0 = zeros(length(meta.yG))
  val = obj(x0)
  return isa(val, Number) && val == 0.0
end

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

  ncc::Int   # number of complementarity constraints
  Glin::Vector{Int}
  Gnlin::Int
  Gnln::Vector{Int}
  Gnnln::Int
  Hlin::Vector{Int}
  Hnlin::Int
  Hnln::Vector{Int}
  Hnnln::Int
  yG::S      # initial Lagrange multipliers
  yH::S      # initial Lagrange multipliers
  lccG::S    # vector of constraint lower bounds of the complementarity constraint
  lccH::S    # vector of constraint upper bounds of the complementarity constraint
  nnzjG::Int
  Glin_nnzj::Int
  Gnln_nnzj::Int
  nnzjH::Int
  Hlin_nnzj::Int
  Hnln_nnzj::Int
end

function MPCCModelMeta{T,S}(
  nvar,
  ncc;
  Glin = Int[],
  Hlin = Int[],
  yG::S = fill!(S(undef, ncc), zero(T)),
  yH::S = fill!(S(undef, ncc), zero(T)),
  lccG::S = fill!(S(undef, ncc), T(-Inf)),
  lccH::S = fill!(S(undef, ncc), T(Inf)),
  nnzjG = ncc * nvar,
  Glin_nnzj = 0,
  nnzjH = ncc * nvar,
  Hlin_nnzj = 0,
) where {T,S}
  @lencheck ncc lccG lccH yG yH
  Gnlin = length(Glin)
  Gnln = setdiff(1:ncc, Glin)
  Gnnln = length(Gnln)
  Hnlin = length(Hlin)
  Hnln = setdiff(1:ncc, Hlin)
  Hnnln = length(Hnln)

  Gnln_nnzj = nnzjG - Glin_nnzj
  Hnln_nnzj = nnzjH - Hlin_nnzj

  MPCCModelMeta{T,S}(
    ncc,
    Glin,
    Gnlin,
    Gnln,
    Gnnln,
    Hlin,
    Hnlin,
    Hnln,
    Hnnln,
    yG,
    yH,
    lccG,
    lccH,
    nnzjG,
    Glin_nnzj,
    Gnln_nnzj,
    nnzjH,
    Hlin_nnzj,
    Hnln_nnzj,
  )
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
