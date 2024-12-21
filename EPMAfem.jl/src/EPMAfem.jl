module EPMAfem

using ConcreteStructs
using Gridap
using CUDA
using SparseArrays
using LinearAlgebra
using HCubature

include("space_dimensions.jl")
import EPMAfem.Dimensions

include("spherical_harmonics_model/spherical_harmonics.jl")
using EPMAfem.SphericalHarmonicsModels

include("space_model/space.jl")
using EPMAfem.SpaceModels

include("abstracttypes.jl")
include("utils.jl")
include("pnequations.jl")
include("pnmodel.jl")
# include("pnvector.jl")

include("pnsystem.jl")
include("pndiscretization.jl")


end # module EPMAfem
