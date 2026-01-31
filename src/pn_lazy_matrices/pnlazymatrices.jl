module PNLazyMatrices

using LinearAlgebra
using LinearAlgebra: inv!
using CUDA
using ConcreteStructs
using Krylov
using Adapt

using EPMAfem.BlockDiagonals

include("pnmatrixstructure.jl")
include("pnmatrices.jl")
include("pnlazymatrix.jl")
include("pnscalar.jl")
include("pnresizematrix.jl")
include("pntransposematrix.jl")
include("pnkronmatrix.jl")
include("pnsummatrix.jl")
include("pnprodmatrix.jl")
include("pncachedmatrix.jl")
include("pnblockmatrix.jl")
include("pninversematrix.jl")
include("pndiagmatrix.jl")

include("pnworkspace.jl")
include("pnmatrixinterface.jl")
include("pnmatrixsimplify.jl")
include("adapt.jl")
include("pnmatrixio.jl")

export AbstractLazyMatrix, AbstractLazyMatrixOrTranspose, LazyScalar, LazyMatrix
export lazy, blockmatrix, kron_AXB, materialize, cache, create_workspace, required_workspace, mul_with!, materialize_with, unlazy
export schur_complement, diagonal

end
