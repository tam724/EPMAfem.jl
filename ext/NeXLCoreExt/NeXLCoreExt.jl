module NeXLCoreExt

using EPMAfem
using EPMAfem.SpaceModels
using EPMAfem.SphericalHarmonicsModels
using EPMAfem.Gridap
using StaticArrays
using ConcreteStructs
using NeXLCore.Unitful
using NeXLCore
using HCubature
using LegendrePolynomials
using LinearAlgebra
using Interpolations

include("dim_basis.jl")
include("scattering_approximation.jl")
include("epmaequations.jl")
include("epmamodel.jl")


end
