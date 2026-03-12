#stdlib
using LinearAlgebra, SparseArrays, Printf, Test
# JSO
using ADNLPModels, JuMP, LinearOperators, NLPModels, NLPModelsJuMP
# Stopping
using Stopping
# This package
using MPCC

const _mpcc_stopping_ext = Base.get_extension(MPCC, :MPCCStoppingExt)

@testset "MPCCStopping extension loaded" begin
  @test _mpcc_stopping_ext !== nothing
end

const MPCCStoppingExt = _mpcc_stopping_ext === nothing ?
  error("MPCCStoppingExt extension could not be loaded; MPCCStopping-related tests cannot run.") :
  _mpcc_stopping_ext
const MPCCAtX = MPCCStoppingExt.MPCCAtX
const MPCCStopping = MPCCStoppingExt.MPCCStopping
const _init_max_counters_mpcc = MPCCStoppingExt._init_max_counters_mpcc
const SStat = MPCCStoppingExt.SStat
const MStat = MPCCStoppingExt.MStat
const CStat = MPCCStoppingExt.CStat
const WStat = MPCCStoppingExt.WStat
include("problems/bard1.jl")
include("problems/ex1.jl")
include("problems/ex1bd.jl")
include("problems/ex2.jl")
include("problems/ex3.jl")

include("rosenbrock.jl")

@testset "MPCCMeta tests" begin
  test_meta = MPCCModelMeta(6, 0)

  @test test_meta.ncc == 0             # number of complementarity constraints
  @test test_meta.yG == zeros(0)    # initial Lagrange multipliers
  @test test_meta.yH == zeros(0)    # initial Lagrange multipliers
  @test test_meta.lccG == zeros(0)   # vector of constraint lower bounds of the complementarity constraint
  @test test_meta.lccH == zeros(0)   # vector of constraint upper bounds of the complementarity constraint
  @test test_meta.nnzjG == 0             # number of complementarity constraints
  @test test_meta.nnzjH == 0             # number of complementarity constraints
  @test complementarity_constrained(test_meta) == false
end

@testset "MPCCCounters tests" begin
  try_counters = MPCCCounters()

  @test getfield(try_counters, :neval_consG) == 0
  @test getfield(try_counters, :neval_consH) == 0
  @test getfield(try_counters, :neval_jacG) == 0
  @test getfield(try_counters, :neval_jacH) == 0
  @test getfield(try_counters, :neval_jGprod) == 0
  @test getfield(try_counters, :neval_jHprod) == 0
  @test getfield(try_counters, :neval_jGtprod) == 0
  @test getfield(try_counters, :neval_jHtprod) == 0
  @test getfield(try_counters, :neval_hessG) == 0
  @test getfield(try_counters, :neval_hessH) == 0
  @test getfield(try_counters, :neval_hGprod) == 0
  @test getfield(try_counters, :neval_hHprod) == 0

  reset!(try_counters)
end

@testset "ADMPCC tests I: no nonlinear constraints" begin
  f = x -> sum(x)
  x0 = ones(6)
  G(x) = [x[1]; x[3]]
  H(x) = [x[2]; x[4]]
  lccg, lcch = zeros(2), zeros(2)
  lvar, uvar = fill(-10.0, size(x0)), fill(10.0, size(x0))
  admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, lvar = lvar, uvar = uvar)

  @test obj(admpcc, admpcc.meta.x0) == 6
  @test grad(admpcc, admpcc.meta.x0) == ones(6)

  @test consG(admpcc, admpcc.meta.x0) == [1, 1]
  @test consH(admpcc, admpcc.meta.x0) == [1, 1]

  @test jacG(admpcc, admpcc.meta.x0) == [1 0 0 0 0 0; 0 0 1 0 0 0]
  @test jGprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jGtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hessG(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG) == zeros(6, 6)

  @test jacH(admpcc, admpcc.meta.x0) == [0 1 0 0 0 0; 0 0 0 1 0 0]
  @test jHprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jHtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hessH(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH) == zeros(6, 6)

  y = vcat(admpcc.cc_meta.yG, admpcc.cc_meta.yH)
  @test hess(admpcc, admpcc.meta.x0, obj_weight = 0.0) == zeros(6, 6)
  @test hprod(admpcc, admpcc.meta.x0, admpcc.meta.x0, obj_weight = 0.0) == zeros(6)
  @test hess(admpcc, admpcc.meta.x0, y) == zeros(6, 6)
  @test hprod(admpcc, admpcc.meta.x0, y, admpcc.meta.x0) == zeros(6)
end

@testset "ADMPCC tests I: with nonlinear constraints" begin
  f = x -> sum(x)
  x0 = ones(6)
  G(x) = [x[1]; x[3]]
  H(x) = [x[2]; x[4]]
  lccg, lcch = zeros(2), zeros(2)
  lvar, uvar = fill(-10.0, size(x0)), fill(10.0, size(x0))
  c = x -> [x[1]]
  lcon, ucon = zeros(1), zeros(1)
  admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, c, lcon, ucon, lvar = lvar, uvar = uvar)

  @test obj(admpcc, admpcc.meta.x0) == 6
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
  @test hessG(admpcc, admpcc.meta.x0, admpcc.cc_meta.yG) == zeros(6, 6)

  @test jacH(admpcc, admpcc.meta.x0) == [0 1 0 0 0 0; 0 0 0 1 0 0]
  @test jHprod(admpcc, admpcc.meta.x0, zeros(6)) == zeros(2)
  @test jHtprod(admpcc, admpcc.meta.x0, zeros(2)) == zeros(6)
  @test hHprod(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH, admpcc.meta.x0) == zeros(6)
  @test hessH(admpcc, admpcc.meta.x0, admpcc.cc_meta.yH) == zeros(6, 6)

  y = vcat(admpcc.meta.y0, admpcc.cc_meta.yG, admpcc.cc_meta.yH)
  @test hess(admpcc, admpcc.meta.x0, obj_weight = 0.0) == zeros(6, 6)
  @test hprod(admpcc, admpcc.meta.x0, admpcc.meta.x0, obj_weight = 0.0) == zeros(6)
  @test hess(admpcc, admpcc.meta.x0, y) == zeros(6, 6)
  @test hprod(admpcc, admpcc.meta.x0, y, admpcc.meta.x0) == zeros(6)
end

@testset "NLMPCC tests" begin
  f = x -> sum(x)
  x0 = ones(2)
  G(x) = [x[1]]
  H(x) = [x[2]]
  lccg, lcch = zeros(1), zeros(1)
  lvar, uvar = fill(-10.0, size(x0)), fill(10.0, size(x0))
  c = x -> [x[1]]
  lcon, ucon = zeros(1), zeros(1)
  admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, c, lcon, ucon, lvar = lvar, uvar = uvar)
  nlp = NLMPCC(admpcc)

  @test nlp.mod.cc_meta.ncc == 1

  @test nlp.meta.nvar == 2
  @test nlp.meta.lvar == [-Inf, -Inf]
  @test nlp.meta.uvar == [Inf, Inf]

  @test nlp.meta.ncon == 4
  @test nlp.meta.lcon == [0.0, -0.0, -0.0, -Inf]
  @test nlp.meta.ucon == [0.0, Inf, Inf, 0.0]

  @test obj(nlp, nlp.meta.x0) == 2
  @test grad(nlp, nlp.meta.x0) == grad(admpcc, x0)
  #@test hess(nlp, nlp.meta.x0) == zeros(2, 2)
  #@test hess(nlp, nlp.meta.x0, obj_weight = 0.0) == zeros(2, 2)

  t0 = ones(2)
  @test cons(nlp, t0) == [1, 1, 1, 1]
  @test jac(nlp, t0) == [1 0; 1 0; 0 1; 1 1]
  @test jac_coord(nlp, t0) == jac(nlp, t0)[:]
  @test jac_structure(nlp) == ([1, 2, 3, 4, 1, 2, 3, 4], [1, 1, 1, 1, 2, 2, 2, 2])
  @test jprod(nlp, t0, zeros(2)) == zeros(4)
  @test jtprod(nlp, t0, zeros(4)) == zeros(2)
  #@test hess(nlp, t0, ones(4)) == [0.0 1.0; 1.0 0.0]
  #@test length(hess_coord(nlp, t0)) == 3
  #@test length(hess_coord(nlp, t0, ones(4))) == 3
  #@test hess_structure(nlp) == ([1, 2, 2], [1, 1, 2])
  @test hprod(nlp, t0, ones(4), nlp.meta.x0) == [2.0, 2.0]
  @test hprod(nlp, t0, nlp.meta.x0) == zeros(2)

  @test viol(nlp, t0) == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
end

@testset "Sparse Jacobian-Hessian Operator Tests" begin
  mpcc = bard1()
  x = copy(mpcc.meta.x0)

  J = jac(mpcc, x)
  J_rows, J_cols = jac_structure(mpcc)
  J_vals = jac_coord(mpcc, x)
  @test J isa SparseMatrixCSC
  @test J == sparse(J_rows, J_cols, J_vals, mpcc.meta.ncon, mpcc.meta.nvar)

  JG = jacG(mpcc, x)
  JG_rows, JG_cols = jacG_structure(mpcc)
  JG_vals = jacG_coord(mpcc, x)
  @test JG isa SparseMatrixCSC
  @test JG == sparse(JG_rows, JG_cols, JG_vals, mpcc.cc_meta.ncc, mpcc.meta.nvar)

  JH = jacH(mpcc, x)
  JH_rows, JH_cols = jacH_structure(mpcc)
  JH_vals = jacH_coord(mpcc, x)
  @test JH isa SparseMatrixCSC
  @test JH == sparse(JH_rows, JH_cols, JH_vals, mpcc.cc_meta.ncc, mpcc.meta.nvar)

  y = vcat(mpcc.meta.y0, mpcc.cc_meta.yG, mpcc.cc_meta.yH)
  H = hess(mpcc, x, y)
  H_rows, H_cols = hess_structure(mpcc)
  H_vals = hess_coord(mpcc, x, y)
  @test H isa Symmetric
  @test H.data isa SparseMatrixCSC
  @test H == Symmetric(sparse(H_rows, H_cols, H_vals, mpcc.meta.nvar, mpcc.meta.nvar), :L)
  @test Matrix(H) ≈ Diagonal([2.0, 8.0, 0.0, 0.0, 0.0])

  HG = hessG(mpcc, x, mpcc.cc_meta.yG)
  HG_rows, HG_cols = hessG_structure(mpcc)
  HG_vals = hessG_coord(mpcc, x, mpcc.cc_meta.yG)
  @test HG isa Symmetric
  @test HG.data isa SparseMatrixCSC
  @test HG == Symmetric(sparse(HG_rows, HG_cols, HG_vals, mpcc.meta.nvar, mpcc.meta.nvar), :L)

  HH = hessH(mpcc, x, mpcc.cc_meta.yH)
  HH_rows, HH_cols = hessH_structure(mpcc)
  HH_vals = hessH_coord(mpcc, x, mpcc.cc_meta.yH)
  @test HH isa Symmetric
  @test HH.data isa SparseMatrixCSC
  @test HH == Symmetric(sparse(HH_rows, HH_cols, HH_vals, mpcc.meta.nvar, mpcc.meta.nvar), :L)
end

@testset "MPCCAtX tests" begin
  state = MPCCAtX(zeros(10), zeros(0), cGx = [0.0], cHx = [1.0])

  @test state.x == zeros(10)
  @test isnan(state.fx)
  @test state.gx == zeros(0)
  @test state.Hx == zeros(0, 0)
  @test state.mu == zeros(0)
  @test state.cx == zeros(0)
  @test state.Jx == zeros(0, 0)

  @test state.lambda == zeros(0)
  @test isnan(state.current_time)

  # On vérifie que la fonction update! fonctionne
  update!(state, x = ones(10), fx = 1.0, gx = ones(10))
  update!(state, lambda = ones(10), current_time = 1.0)
  update!(state, Hx = ones(10, 10), mu = ones(10), cx = ones(10), Jx = ones(10, 10))

  @test (false in (state.x .== 1.0)) == false #assez bizarre comme test...
  @test state.fx == 1.0
  @test (false in (state.gx .== 1.0)) == false
  @test (false in (state.Hx .== 1.0)) == false
  @test state.mu == ones(10)
  @test state.cx == ones(10)
  @test (false in (state.Jx .== 1.0)) == false
  @test (false in (state.lambda .== 1.0)) == false
  @test state.current_time == 1.0

  reinit!(state)
  @test state.x == ones(10)
  @test isnan(state.fx)
  reinit!(state, x = zeros(10))
  @test state.x == zeros(10)
  @test isnan(state.fx)

end

@testset "MPCCStopping tests" begin
  include("test-mpcc-stopping.jl")
end
