using LinearAlgebra, LinearOperators, SparseArrays, Printf, Test

using MPCC

using JuMP
using NLPModelsJuMP
using NLPModels

include("problems.jl")
include("rosenbrock.jl")

printstyled("MPCCMeta tests... ")
include("test-mpcc-meta.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("MPCCCounters tests... ")
include("test-mpcc-counters.jl")
printstyled("passed ✓ \n", color = :green)
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
