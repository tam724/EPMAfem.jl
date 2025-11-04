module SphericalHarmonicsModels

using ConcreteStructs
using ComponentArrays
using Serialization
using LinearAlgebra
using Gridap: VectorValue

using LegendrePolynomials
using SphericalHarmonics

using HCubature
using Lebedev
using EPMAfem.Dimensions

include("spherical_harmonic_moments.jl")
include("spherical_harmonics_model.jl")
include("spherical_harmonic_matrices.jl")
include("spherical_harmonics_transport.jl")
include("spherical_harmonics_boundary.jl")
include("spherical_harmonic_vectors.jl")

export AbstractSphericalHarmonicsModel, EOSphericalHarmonicsModel, EEEOSphericalHarmonicsModel, minus, plus
export assemble_bilinear, assemble_linear

end
