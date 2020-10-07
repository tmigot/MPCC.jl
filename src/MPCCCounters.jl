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

  neval_obj     :: Int  # Number of objective evaluations.
  neval_grad    :: Int  # Number of objective gradient evaluations.
  neval_cons    :: Int  # Number of constraint vector evaluations.
  neval_consG   :: Int  # Number of constraint vector evaluations of G(x).
  neval_consH   :: Int  # Number of constraint vector evaluations of H(x).
  neval_jcon    :: Int  # Number of individual constraint evaluations.
  neval_jgrad   :: Int  # Number of individual constraint gradient evaluations.
  neval_jac     :: Int  # Number of constraint Jacobian evaluations.
  neval_jacG    :: Int  # Number of constraint Jacobian evaluations of G(x).
  neval_jacH    :: Int  # Number of constraint Jacobian evaluations of H(x).
  neval_jprod   :: Int  # Number of Jacobian-vector products.
  neval_jGprod  :: Int  # Number of Jacobian-vector products.
  neval_jHprod  :: Int  # Number of Jacobian-vector products.
  neval_jtprod  :: Int  # Number of transposed Jacobian-vector products.
  neval_jGtprod :: Int  # Number of transposed Jacobian-vector products.
  neval_jHtprod :: Int  # Number of transposed Jacobian-vector products.
  neval_hess    :: Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hessG   :: Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hessH   :: Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hprod   :: Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_hGprod  :: Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_hHprod  :: Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_jhprod  :: Int  # Number of individual constraint Hessian-vector products.

  function MPCCCounters()
    return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  end
end

# simple default API for retrieving counters
for counter in fieldnames(MPCCCounters)
  @eval begin
    """
        $($counter)(nlp)
    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nlp :: AbstractMPCCModel) = nlp.counters.$counter
    export $counter
  end
end

"""
    increment!(nlp, s)
Increment counter `s` of problem `nlp`.
"""
function increment!(nlp :: AbstractMPCCModel, s :: Symbol)
  setfield!(nlp.counters, s, getfield(nlp.counters, s) + 1)
end

"""
    decrement!(nlp, s)
Increment counter `s` of problem `nlp`.
"""
function decrement!(nlp :: AbstractMPCCModel, s :: Symbol)
  if s in fieldnames(MPCCCounters)
    setfield!(nlp.counters, s, getfield(nlp.counters, s) - 1)
  end
end

"""
    sum_counters(counters)
Sum all counters of `counters`.
"""
sum_counters(c :: MPCCCounters) = sum(getfield(c, x) for x in fieldnames(MPCCCounters))

"""
    sum_counters(nlp)
Sum all counters of problem `nlp`.
"""
sum_counters(nlp :: AbstractMPCCModel) = sum_counters(nlp.counters)

"""
    reset!(counters)
Reset evaluation counters
"""
function reset!(counters :: MPCCCounters)
  for f in fieldnames(MPCCCounters)
    setfield!(counters, f, 0)
  end
  return counters
end

"""
    reset!(nlp)
Reset evaluation count in `nlp`
"""
function reset!(nlp :: AbstractMPCCModel)
  reset!(nlp.counters)
  return nlp
end
