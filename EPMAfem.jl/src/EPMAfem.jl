module EPMAfem

using ConcreteStructs
using Gridap
using CUDA
using SparseArrays
using LinearAlgebra
using HCubature
using Distributions
using Krylov
using UnsafeArrays

include("special_matrices/sparse3tensor.jl")
import EPMAfem.Sparse3Tensor
include("special_matrices/blockedmatrices.jl")
import EPMAfem.BlockedMatrices

include("space_dimensions.jl")
import EPMAfem.Dimensions

include("spherical_harmonics_model/spherical_harmonics.jl")
using EPMAfem.SphericalHarmonicsModels

include("space_model/space.jl")
using EPMAfem.SpaceModels

include("abstracttypes.jl")
include("utils.jl")
include("pnequations.jl")
include("pnarchitecture.jl")
include("pnmodel.jl")
include("pnproblem.jl")
include("pnindex.jl")
include("pniterators.jl")
include("pnvector.jl")
# include("pnderivativevector.jl")

include("pndiscretization.jl")
include("systems/pnsystems.jl")


end # module EPMAfem
