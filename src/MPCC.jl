module MPCC

using LinearAlgebra, LinearOperators, SparseArrays, FastClosures
using NLPModels, Printf

using Test

import NLPModels:
    obj,
    grad,
    grad!,
    objgrad,
    objgrad!,
    objcons,
    objcons!,
    cons,
    cons!,
    jth_con,
    jth_congrad,
    jth_congrad!,
    jth_sparse_congrad,
    jac_structure!,
    jac_structure,
    jac_coord!,
    jac_coord,
    jac,
    jprod,
    jprod!,
    jtprod,
    jtprod!,
    jac_op,
    jac_op!,
    jth_hprod,
    jth_hprod!,
    ghjvprod,
    ghjvprod!,
    hess_structure!,
    hess_structure,
    hess_coord!,
    hess_coord,
    hess,
    hess_op,
    hess_op!,
    hprod,
    hprod!,
    sum_counters,
    reset!,
    increment!

using NLPModels: @lencheck, @rangecheck

"""
Base type for an optimization model with degenerate constraints.

    min f(x)
    l <= x <= u
    lb <= c(x) <= ub
    0 <= G(x) _|_ H(x) >= 0
"""
abstract type AbstractMPCCModel{T,S} <: AbstractNLPModel{T,S} end

include("MPCCModelMeta.jl")

export AbstractMPCCModelMeta, MPCCModelMeta, complementarity_constrained

include("MPCCCounters.jl")

export MPCCCounters, increment!, reset!, sum_counters, decrement!

include("AbstractMPCC.jl")

export AbstractMPCCModel,
    obj,
    grad,
    grad!,
    objgrad,
    objgrad!,
    objcons,
    objcons!,
    cons,
    cons!,
    cons_nl,
    cons_nl!,
    consG,
    consG!,
    consH,
    consH!,
    viol,
    viol!,
    jth_con,
    jth_congrad,
    jth_congrad!,
    jth_sparse_congrad,
    jac_structure!,
    jac_structure,
    jac_coord!,
    jac_coord,
    jac,
    jprod,
    jprod!,
    jtprod,
    jtprod!,
    jac_op,
    jac_op!,
    jac_nl_structure!,
    jac_nl_structure,
    jac_nl_coord!,
    jac_nl_coord,
    jac_nl,
    jnlprod,
    jnlprod!,
    jnltprod,
    jnltprod!,
    jac_nl_op,
    jac_nl_op!,
    jacG_structure!,
    jacG_structure,
    jacG_coord!,
    jacG_coord,
    jacG,
    jGprod,
    jGprod!,
    jGtprod,
    jGtprod!,
    jacG_op,
    jacG_op!,
    jacH_structure!,
    jacH_structure,
    jacH_coord!,
    jacH_coord,
    jacH,
    jHprod,
    jHprod!,
    jHtprod,
    jHtprod!,
    jacH_op,
    jacH_op!,
    jth_hprod,
    jth_hprod!,
    ghjvprod,
    ghjvprod!,
    hess_structure!,
    hess_structure,
    hess_coord!,
    hess_coord,
    hessG_structure!,
    hessG_structure,
    hessH_structure!,
    hessH_structure,
    hessG_coord!,
    hessG_coord,
    hessH_coord!,
    hessH_coord,
    hess,
    hessG,
    hessH,
    hprod,
    hprod!,
    hGprod,
    hGprod!,
    hHprod,
    hHprod!,
    hess_op,
    hess_op!,
    hessG_op,
    hessG_op!,
    hessH_op,
    hessH_op!

export neval_obj, neval_grad, neval_hess, neval_hprod, neval_cons, neval_cons_lin, neval_cons_nln, neval_jac, neval_jac_lin, neval_jac_nln

using ADNLPModels, ForwardDiff

include("ADMPCC.jl")

export ADMPCCModel

include("MPCCNLPs.jl")

export MPCCNLPs #, jacl, jac_actif

include("NLMPCC.jl")

export NLMPCC

using Stopping

import Stopping: reinit!, update!, _init_field

include("MPCCState.jl")

export MPCCAtX

include("MPCCStopping.jl")

export MPCCStopping, _init_max_counters_mpcc, SStat, MStat, CStat, WStat

end #end of module
