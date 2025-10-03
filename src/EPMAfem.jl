module EPMAfem

using ConcreteStructs
using Gridap
using CUDA
using KernelAbstractions
using GPUArrays
using SparseArrays
using LinearAlgebra
using HCubature
using Distributions
using Krylov
using UnsafeArrays
using Zygote
using ChainRulesCore
using Adapt

include("hacky_overwrites.jl")

include("special_matrices/sparse3tensor.jl")
using EPMAfem.Sparse3Tensor
include("special_matrices/blockedmatrices.jl")
using EPMAfem.BlockedMatrices

include("special_matrices/twodiagonalmatrix.jl")

include("pn_lazy_matrices/pnlazymatrices.jl")
using EPMAfem.PNLazyMatrices

include("space_dimensions.jl")
using EPMAfem.Dimensions

include("spherical_harmonics_model/spherical_harmonics.jl")
using EPMAfem.SphericalHarmonicsModels

include("space_model/space.jl")
using EPMAfem.SpaceModels


PNLazyMatrices.has_batched_mul!(A::TwoDiagonalMatrix) = true
PNLazyMatrices.has_batched_mul!(A::Transpose{<:Number, <:TwoDiagonalMatrix}) = true

include("utils.jl")

include("abstracttypes.jl")
include("pnequations.jl")
include("pnequations_filtered.jl")
include("pnarchitecture.jl")
include("pnmodel.jl")
include("pnproblem.jl")
include("pnsource_pnboundary.jl")

include("pnindex.jl")
include("pniterators.jl")
include("pnabsorption.jl")
include("pnvector.jl")
include("pnderivativevector.jl")
include("pnprobes.jl")

include("pndiscretization.jl")
include("systems/pnsystems.jl")
include("epmaproblem.jl")

include("monochrom_pn/monochrom_pn.jl")
include("degenerate_pn/degenerate_pn.jl")


end # module EPMAfem
