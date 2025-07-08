module PNLazyMatrices

using LinearAlgebra
using CUDA
using ConcreteStructs

include("pnmatrices.jl")
include("pntransposematrix.jl")
include("pnkronmatrix.jl")
include("pnsummatrix.jl")
include("pnprodmatrix.jl")
include("pnblockmatrix.jl")
include("pncachedmatrix.jl")
include("pnresizematrix.jl")
include("pnmatrixinterface.jl")
include("pnworkspace.jl")

export lazy, blockmatrix, kron_AXB, materialize, cache, create_workspace, required_workspace, mul_with!, materialize_with, unlazy

end
