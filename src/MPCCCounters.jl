#####################################################################################
# MPCCMCounters type ; based on NLPModels Counters:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl
#
# Problem both NLPModels and MPCC export neval_hess,...
#####################################################################################

"""
MPCCCounters

Initialization: `MPCCCounters()`
"""
mutable struct MPCCCounters

    neval_consG::Int  # Number of constraint vector evaluations of G(x).
    neval_consH::Int  # Number of constraint vector evaluations of H(x).
    neval_jacG::Int  # Number of constraint Jacobian evaluations of G(x).
    neval_jacH::Int  # Number of constraint Jacobian evaluations of H(x).
    neval_jGprod::Int  # Number of Jacobian-vector products.
    neval_jHprod::Int  # Number of Jacobian-vector products.
    neval_jGtprod::Int  # Number of transposed Jacobian-vector products.
    neval_jHtprod::Int  # Number of transposed Jacobian-vector products.
    neval_hessG::Int  # Number of Lagrangian/objective Hessian evaluations.
    neval_hessH::Int  # Number of Lagrangian/objective Hessian evaluations.
    neval_hGprod::Int  # Number of Lagrangian/objective Hessian-vector products.
    neval_hHprod::Int  # Number of Lagrangian/objective Hessian-vector products.

    function MPCCCounters()
        return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    end
end

# simple default API for retrieving counters
for counter in fieldnames(MPCCCounters)
    @eval begin
        """
            $($counter)(nlp)
        Get the number of `$(split("$($counter)", "_")[2])` evaluations.
        """
        $counter(nlp::AbstractMPCCModel) = nlp.counters.$counter
        export $counter
    end
end

"""
    increment!(nlp, s)
Increment counter `s` of problem `nlp`.
"""
function increment!(nlp::AbstractMPCCModel, s::Symbol)
    if s in fieldnames(MPCCCounters)
        setfield!(nlp.cc_counters, s, getfield(nlp.cc_counters, s) + 1)
    else
        setfield!(nlp.counters, s, getfield(nlp.counters, s) + 1)
    end
end

"""
    decrement!(nlp, s)
Increment counter `s` of problem `nlp`.
"""
function decrement!(nlp::AbstractMPCCModel, s::Symbol)
    if s in fieldnames(MPCCCounters)
        setfield!(nlp.cc_counters, s, getfield(nlp.cc_counters, s) - 1)
    else
        setfield!(nlp.counters, s, getfield(nlp.counters, s) - 1)
    end
end

"""
    reset!(counters)
Reset evaluation counters
"""
function reset!(counters::MPCCCounters)
    for f in fieldnames(MPCCCounters)
        setfield!(counters, f, 0)
    end
    return counters
end

"""
    reset!(nlp)
Reset evaluation count in `nlp`
"""
function reset!(nlp::AbstractMPCCModel)
    reset!(nlp.cc_counters)
    return nlp
end
