using LinearAlgebra, LinearOperators, SparseArrays, Printf, Test

using MPCC

using JuMP
using NLPModelsJuMP
using NLPModels

include("problems.jl")
include("rosenbrock.jl")

@testset "MPCCMeta tests" begin
  test_meta = MPCCModelMeta()

  @test test_meta.ncc   == 0             # number of complementarity constraints
  @test test_meta.yG   == zeros(0)    # initial Lagrange multipliers
  @test test_meta.yH   == zeros(0)    # initial Lagrange multipliers
  @test test_meta.lccG  == zeros(0)   # vector of constraint lower bounds of the complementarity constraint
  @test test_meta.lccH  == zeros(0)   # vector of constraint upper bounds of the complementarity constraint
  @test complementarity_constrained(test_meta) == false
end

@testset "MPCCCounters tests" begin
  try_counters = MPCCCounters()

  @test getfield(try_counters, :neval_consG)   == 0
  @test getfield(try_counters, :neval_consH)   == 0
  @test getfield(try_counters, :neval_jacG)    == 0
  @test getfield(try_counters, :neval_jacH)    == 0
  @test getfield(try_counters, :neval_jGprod)  == 0
  @test getfield(try_counters, :neval_jHprod)  == 0
  @test getfield(try_counters, :neval_jGtprod) == 0
  @test getfield(try_counters, :neval_jHtprod) == 0
  @test getfield(try_counters, :neval_hessG)   == 0
  @test getfield(try_counters, :neval_hessH)   == 0
  @test getfield(try_counters, :neval_hGprod)  == 0
  @test getfield(try_counters, :neval_hHprod)  == 0

  @test sum_counters(try_counters) == 0

  reset!(try_counters)
  @test sum_counters(try_counters) == 0

end

#=
printstyled("ADMPCC tests... ")
include("run_test_admpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("MPCCNLPs tests... ")
include("run_test_mpccnlps.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("NLMPCC tests... ")
include("test-mpcc-nlmpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("Bilevel Programming MPCC tests... ")
include("run_test_bpmpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("Ampl MPCC tests... ")
include("run_test_mpccampl.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("MPCCAtX tests... ")
include("test-mpcc-state.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("MPCCStopping tests... ")
include("test-mpcc-stopping.jl")
printstyled("passed ✓ \n", color = :green)
=#
