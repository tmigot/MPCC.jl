using LinearAlgebra, LinearOperators, SparseArrays, Printf, Test

using ADNLPModels
using MPCC

using JuMP
using NLPModelsJuMP
using NLPModels

include("problems.jl")
include("rosenbrock.jl")

@testset "MPCCMeta tests" begin
  test_meta = MPCCModelMeta(6, 0)

  @test test_meta.ncc   == 0             # number of complementarity constraints
  @test test_meta.yG   == zeros(0)    # initial Lagrange multipliers
  @test test_meta.yH   == zeros(0)    # initial Lagrange multipliers
  @test test_meta.lccG  == zeros(0)   # vector of constraint lower bounds of the complementarity constraint
  @test test_meta.lccH  == zeros(0)   # vector of constraint upper bounds of the complementarity constraint
  @test test_meta.nnzjG   == 0             # number of complementarity constraints
  @test test_meta.nnzjH   == 0             # number of complementarity constraints
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

  reset!(try_counters)
end

@testset "ADMPCC tests I: no nonlinear constraints" begin
  f = x -> sum(x)
  x0 = ones(6)
  G(x) = [x[1];x[3]]
  H(x) = [x[2];x[4]]
  lccg, lcch = zeros(2), zeros(2)
  lvar, uvar = fill(-10.0,size(x0)), fill(10.0,size(x0))
  admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, lvar = lvar, uvar = uvar)

  @test obj(admpcc, admpcc.meta.x0)  == 6
  @test grad(admpcc, admpcc.meta.x0) == ones(6)

  @test consG(admpcc, admpcc.meta.x0) == [1, 1]
  @test consH(admpcc, admpcc.meta.x0) == [1, 1]

  @test jacG(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0; 0 0 1 0 0 0]
  @test jGprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jGtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hessG(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG) == zeros(6,6)

  @test jacH(admpcc, admpcc.meta.x0) == [0 1 0 0 0 0; 0 0 0 1 0 0]
  @test jHprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jHtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hessH(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH) == zeros(6,6)

  y = vcat(admpcc.cc_meta.yG, admpcc.cc_meta.yH)
  @test hess(admpcc, admpcc.meta.x0, obj_weight = 0.0) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, admpcc.meta.x0, obj_weight = 0.0) == zeros(6)
  @test hess(admpcc, admpcc.meta.x0, y) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, y, admpcc.meta.x0) == zeros(6)
end

@testset "ADMPCC tests I: with nonlinear constraints" begin
  f = x -> sum(x)
  x0 = ones(6)
  G(x) = [x[1];x[3]]
  H(x) = [x[2];x[4]]
  lccg, lcch = zeros(2), zeros(2)
  lvar, uvar = fill(-10.0, size(x0)), fill(10.0,size(x0))
  c = x -> [x[1]]
  lcon, ucon = zeros(1), zeros(1)
  admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, c, lcon, ucon, lvar = lvar, uvar = uvar)

  @test obj(admpcc, admpcc.meta.x0)  == 6
  @test grad(admpcc, admpcc.meta.x0) == ones(6)

  @test cons(admpcc, admpcc.meta.x0) == [1]
  @test consG(admpcc, admpcc.meta.x0) == [1, 1]
  @test consH(admpcc, admpcc.meta.x0) == [1, 1]

  @test jac(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0]
  @test jprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(1)
  @test jtprod(admpcc, admpcc.meta.x0, zeros(1)) == zeros(6)

  @test jacG(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0; 0 0 1 0 0 0]
  @test jGprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jGtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hGprod(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG, admpcc.meta.x0) == zeros(6)
  @test hessG(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG) == zeros(6,6)

  @test jacH(admpcc, admpcc.meta.x0) == [0 1 0 0 0 0; 0 0 0 1 0 0]
  @test jHprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jHtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hHprod(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH, admpcc.meta.x0) == zeros(6)
  @test hessH(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH) == zeros(6,6)

  y = vcat(admpcc.meta.y0, admpcc.cc_meta.yG, admpcc.cc_meta.yH)
  @test hess(admpcc, admpcc.meta.x0, obj_weight = 0.0) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, admpcc.meta.x0, obj_weight = 0.0) == zeros(6)
  @test hess(admpcc, admpcc.meta.x0, y) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, y, admpcc.meta.x0) == zeros(6)
end

@testset "MPCCNLPs tests"  begin
  f = x -> sum(x)
  x0 = ones(6)
  G(x) = [x[1];x[3]]
  H(x) = [x[2];x[4]]
  lccg, lcch = zeros(2), zeros(2)
  lvar, uvar = fill(-10.0, size(x0)), fill(10.0,size(x0))
  c = x -> [x[1]]
  lcon, ucon = zeros(1), zeros(1)
  mp = ADNLPModel(f, x0, c, lcon, ucon, lvar = lvar, uvar = uvar)
  G = ADNLPModel(x -> 0.0, x0, G, lccg, Inf * ones(2))
  H = ADNLPModel(x -> 0.0, x0, H, lcch, Inf * ones(2))
  admpcc = MPCCNLPs(mp, G, H)

  @test obj(admpcc, admpcc.meta.x0)  == 6
  @test grad(admpcc, admpcc.meta.x0) == ones(6)

  @test cons(admpcc, admpcc.meta.x0) == [1]
  @test consG(admpcc, admpcc.meta.x0) == [1, 1]
  @test consH(admpcc, admpcc.meta.x0) == [1, 1]

  @test jac(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0]
  @test jprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(1)
  @test jtprod(admpcc, admpcc.meta.x0, zeros(1)) == zeros(6)

  @test jacG(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0; 0 0 1 0 0 0]
  @test jGprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jGtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hGprod(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG, admpcc.meta.x0) == zeros(6)
  @test hessG(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG) == zeros(6,6)

  @test jacH(admpcc, admpcc.meta.x0) == [0 1 0 0 0 0; 0 0 0 1 0 0]
  @test jHprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jHtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hHprod(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH, admpcc.meta.x0) == zeros(6)
  @test hessH(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH) == zeros(6,6)

  y = vcat(admpcc.meta.y0, admpcc.cc_meta.yG, admpcc.cc_meta.yH)
  @test hess(admpcc, admpcc.meta.x0, obj_weight = 0.0) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, admpcc.meta.x0, obj_weight = 0.0) == zeros(6)
  @test hess(admpcc, admpcc.meta.x0, y) == zeros(6,6)
  @test hprod(admpcc, admpcc.meta.x0, y, admpcc.meta.x0) == zeros(6)
end

#=
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
