module PNLazyMatrices

using LinearAlgebra
using LinearAlgebra: inv!
using CUDA
using ConcreteStructs
using Krylov

include("pnmatrices.jl")
include("pnscalar.jl")
include("pntransposematrix.jl")
include("pnkronmatrix.jl")
include("pnsummatrix.jl")
include("pnprodmatrix.jl")
include("pnblockmatrix.jl")
include("pnresizematrix.jl")
include("pninversematrix.jl")
include("pncachedmatrix.jl")
include("pnworkspace.jl")

include("pnmatrixinterface.jl")

export AbstractLazyMatrix, AbstractLazyMatrixOrTranspose, LazyScalar
export lazy, blockmatrix, kron_AXB, materialize, cache, create_workspace, required_workspace, mul_with!, materialize_with, unlazy
export invalidate_cache!

end
