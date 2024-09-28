# MPCCMCounters type ; based on NLPModels Counters:
# https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/nlp_types.jl

"""
MPCCCounters

Initialization: `MPCCCounters()`
"""
mutable struct MPCCCounters

  neval_consG::Int  # Number of constraint vector evaluations of G(x).
  neval_consG_lin::Int  # Number of linear constraint vector evaluations.
  neval_consG_nln::Int  # Number of nonlinear constraint vector evaluations.
  neval_consH::Int  # Number of constraint vector evaluations of H(x).
  neval_consH_lin::Int  # Number of linear constraint vector evaluations.
  neval_consH_nln::Int  # Number of nonlinear constraint vector evaluations.
  neval_jacG::Int  # Number of constraint Jacobian evaluations of G(x).
  neval_jacG_lin::Int
  neval_jacG_nln::Int
  neval_jacH::Int  # Number of constraint Jacobian evaluations of H(x).
  neval_jacH_lin::Int
  neval_jacH_nln::Int
  neval_jGprod::Int  # Number of Jacobian-vector products.
  neval_jGprod_lin::Int
  neval_jGprod_nln::Int
  neval_jHprod::Int  # Number of Jacobian-vector products.
  neval_jHprod_lin::Int
  neval_jHprod_nln::Int
  neval_jGtprod::Int  # Number of transposed Jacobian-vector products.
  neval_jGtprod_lin::Int
  neval_jGtprod_nln::Int
  neval_jHtprod::Int  # Number of transposed Jacobian-vector products.
  neval_jHtprod_lin::Int
  neval_jHtprod_nln::Int
  neval_hessG::Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hessH::Int  # Number of Lagrangian/objective Hessian evaluations.
  neval_hGprod::Int  # Number of Lagrangian/objective Hessian-vector products.
  neval_hHprod::Int  # Number of Lagrangian/objective Hessian-vector products.

  function MPCCCounters()
    return new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  end
end

# simple default API for retrieving counters
for counter in fieldnames(MPCCCounters)
  @eval begin
    """
        $($counter)(nlp)
    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nlp::AbstractMPCCModel) = nlp.cc_counters.$counter
    export $counter
  end
end

"""
    increment_cc!(nlp, s)
Increment counter `s` of problem `nlp`.
"""
function increment_cc!(nlp::AbstractMPCCModel, s::Symbol)
  setfield!(nlp.cc_counters, s, getfield(nlp.cc_counters, s) + 1)
end

"""
    decrement!(nlp, s)
Decrement counter `s` of problem `nlp`.
"""
function decrement_cc!(nlp::AbstractMPCCModel, s::Symbol)
  setfield!(nlp.cc_counters, s, getfield(nlp.cc_counters, s) - 1)
end

"""
    reset!(counters)
Reset evaluation counters
"""
function NLPModels.reset!(counters::MPCCCounters)
  for f in fieldnames(MPCCCounters)
    setfield!(counters, f, 0)
  end
  return counters
end

"""
    reset!(nlp)
Reset evaluation count in `nlp`
"""
function NLPModels.reset!(nlp::AbstractMPCCModel)
  reset!(nlp.cc_counters)
  return nlp
end

# Adapted from https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/d05c1509ac2041fb39e0a500b1b67cc1b353971a/src/nlp/utils.jl#L119
macro default_cc_counters(Model, inner)
  ex = Expr(:block)
  for foo in fieldnames(Counters) ∪ [:sum_counters]
    push!(ex.args, :(NLPModels.$foo(nlp::$(esc(Model))) = $foo(nlp.$inner)))
  end
  push!(
    ex.args,
    :(NLPModels.increment!(nlp::$(esc(Model)), s::Symbol) = increment!(nlp.$inner, s)),
  )
  push!(
    ex.args,
    :(NLPModels.decrement!(nlp::$(esc(Model)), s::Symbol) = decrement!(nlp.$inner, s)),
  )
  ex
end
