__precompile__()

module MPCC

using LinearAlgebra, LinearOperators, SparseArrays, FastClosures
using NLPModels, Printf

import NLPModels: obj, grad, grad!, objgrad, objgrad!, objcons, objcons!, cons, cons!,
                  jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
                  jac_structure!, jac_structure, jac_coord!, jac_coord,
                  jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!,
                  jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
                  hess_structure!, hess_structure, hess_coord!, hess_coord,
                  hess, hess_op, hess_op!, hprod, hprod!,
                  sum_counters, reset!, increment!

using NLPModels: @lencheck, @rangecheck

"""
Introduce the MPCC :
min f(x)
l <= x <= u
lb <= c(x) <= ub
0 <= G(x) _|_ H(x) >= 0
"""
abstract type AbstractMPCCModel end

#####################################################################################
# MPCCModelMeta type ; based on NLPModelMeta:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl
#####################################################################################
include("MPCCModelMeta.jl")

export AbstractMPCCModelMeta, MPCCModelMeta, print,
       has_bounds, bound_constrained, unconstrained,
       linearly_constrained, equality_constrained,
       inequality_constrained, complementarity_constrained

#####################################################################################
# MPCCMCounters type ; based on NLPModels Counters:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl
#####################################################################################
include("MPCCCounters.jl")

export MPCCCounters, increment!, reset!, sum_counters

include("AbstractMPCC.jl")

export AbstractMPCCModel,
       obj, grad, grad!, objgrad, objgrad!, objcons, objcons!,
       cons, cons!,cons_nl, cons_nl!,consG, consG!,consH, consH!, viol, viol!,
       jth_con, jth_congrad, jth_congrad!, jth_sparse_congrad,
       jac_structure!, jac_structure, jac_coord!, jac_coord,
       jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!,
       jac_nl_structure!, jac_nl_structure, jac_nl_coord!, jac_nl_coord,
       jac_nl, jnlprod, jnlprod!, jnltprod, jnltprod!, jac_nl_op, jac_nl_op!,
       jacG_structure!, jacG_structure, jacG_coord!, jacG_coord,
       jacG, jGprod, jGprod!, jGtprod, jGtprod!, jacG_op, jacG_op!,
       jacH_structure!, jacH_structure, jacH_coord!, jacH_coord,
       jacH, jHprod, jHprod!, jHtprod, jHtprod!, jacH_op, jacH_op!,
       jth_hprod, jth_hprod!, ghjvprod, ghjvprod!,
       hess_structure!, hess_structure, hess_coord!, hess_coord,
       hessG_structure!, hessG_structure, hessH_structure!, hessH_structure,
       hessG_coord!, hessG_coord, hessH_coord!, hessH_coord,
       hess, hessG, hessH,
       hprod, hprod!, hGprod, hGprod!,hHprod, hHprod!,
       hess_op, hess_op!,hessG_op, hessG_op!,hessH_op, hessH_op!

using ForwardDiff

include("ADMPCC.jl")

export ADMPCCModel

include("MPCCNLPs.jl")

export MPCCNLPs, jacl, jac_actif

include("BPMPCC.jl")

export BPMPCCModel

include("NLMPCC.jl")

export NLMPCC

include("MPCCState.jl")

export MPCCAtX, update!, reinit!

include("MPCCStopping.jl")

export MPCCStopping, _init_max_counters, fill_in!, _resources_check!,
       _unbounded_problem_check!, _optimality_check,
       start!, stop!, update_and_start!, update_and_stop!, reinit!, status,
       SStat, MStat, CStat, WStat

end #end of module
