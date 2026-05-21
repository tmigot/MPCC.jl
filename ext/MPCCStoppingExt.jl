module MPCCStoppingExt

using Stopping, MPCC
using NLPModels
using LinearAlgebra
using LinearOperators
using SparseArrays

include("MPCCStopping/MPCCState.jl")

export MPCCAtX

include("MPCCStopping/MPCCStopping.jl")

export MPCCStopping, _init_max_counters_mpcc, SStat, MStat, CStat, WStat

end # end of module
