module NeXLCoreExt

using EPMAfem
using EPMAfem.SpaceModels
using EPMAfem.SphericalHarmonicsModels
using EPMAfem.Gridap
using Dimensionless
using ConcreteStructs
using NeXLCore.Unitful
using NeXLCore
using HCubature
using LegendrePolynomials
using LinearAlgebra
using Interpolations


# some Dimensionless additions
"""
    dimless(quantity_array, basis)

Make an `quantity_array` dimensionless using a dimensional `basis`.
"""
function Dimensionless.dimless(quantity_array::AbstractArray{<:Unitful.AbstractQuantity}, basis::Dimensionless.QuantityDimBasis)
    fac = Dimensionless.fac_dimful(Unitful.unit(eltype(quantity_array)), basis)
    return Unitful.ustrip.(quantity_array) ./ fac
end

include("scattering_approximation.jl")
include("epmaequations.jl")
include("epmamodel.jl")


end