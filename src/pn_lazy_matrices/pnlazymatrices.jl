module PNLazyMatrices

using LinearAlgebra
using LinearAlgebra: inv!
using CUDA
using ConcreteStructs
using Krylov

include("pnmatrices.jl")
include("pntransposematrix.jl")
include("pnkronmatrix.jl")
include("pnsummatrix.jl")
include("pnprodmatrix.jl")
include("pnblockmatrix.jl")
include("pncachedmatrix.jl")
include("pnresizematrix.jl")
include("pnmatrixinterface.jl")
include("pninversematrix.jl")
include("pnworkspace.jl")

export AbstractLazyMatrix, AbstractLazyMatrixOrTranspose
export lazy, blockmatrix, kron_AXB, materialize, cache, create_workspace, required_workspace, mul_with!, materialize_with, unlazy
export invalidate_cache!

end
