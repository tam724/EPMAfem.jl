module Dimensions

using Gridap: VectorValue

# definitions for space dimensions
abstract type SpaceDimension end
struct Z <: SpaceDimension end
struct X <: SpaceDimension end
struct Y <: SpaceDimension end
# dimension(D::Int) = (D in (1, 2, 3)) ? Val(D) : error("dimension D = $(D) must be 1, 2 or 3 (x, y or z)")

# we use minus here to indicate that we refer to the number of dimensions
abstract type SpaceDimensionality end
struct _1D <: SpaceDimensionality end
struct _2D <: SpaceDimensionality end
struct _3D <: SpaceDimensionality end

dimensions(::_1D) = (Z(), )
dimensions(::_2D) = (Z(), X())
dimensions(::_3D) = (Z(), X(), Y())
dimensions() = (Z(), X(), Y())
function dimensionality_type(ND::Int)
    if ND == 1
        return _1D()
    elseif ND == 2
        return _2D()
    elseif ND == 3
        return _3D()
    else
        error("number of dimensions ND=$(ND) must be 1, 2 or 3 (1D, 2D or 3D)")
    end
end

cartesian_unit_vector(::Z, ::_1D) = VectorValue(1.0)

cartesian_unit_vector(::Z, ::_2D) = VectorValue(1.0, 0.0)
cartesian_unit_vector(::X, ::_2D) = VectorValue(0.0, 1.0)

cartesian_unit_vector(::Z, ::_3D) = VectorValue(1.0, 0.0, 0.0)
cartesian_unit_vector(::X, ::_3D) = VectorValue(0.0, 1.0, 0.0)
cartesian_unit_vector(::Y, ::_3D) = VectorValue(0.0, 0.0, 1.0)

extend_3D(x::VectorValue{1}) = VectorValue(x[1], 0.0, 0.0)
extend_3D(x::VectorValue{2}) = VectorValue(x[1], x[2], 0.0)
extend_3D(x::VectorValue{3}) = x

Ωz(Ω::VectorValue{3}) = Ω[1]
Ωx(Ω::VectorValue{3}) = Ω[2]
Ωy(Ω::VectorValue{3}) = Ω[3]

to_Ω(z, x, y) = VectorValue(z, x, y)
from_Ω(Ω) = (; z=Ωz(Ω), x=Ωx(Ω), y=Ωy(Ω))


# see naming convention of https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
function unitsphere_cartesian_to_spherical(Ω::VectorValue{3})
    z, x, y = Ωz(Ω), Ωx(Ω), Ωy(Ω)

    r = sqrt(x*x + y*y + z*z)
    θ = atan(sqrt(x*x + y*y), z)
    ϕ = atan(y, x)
    if !isapprox(r, 1.0)
        @warn "normalizing direction from $r to 1.0"
    end
    return (θ, ϕ)
end

function unitsphere_spherical_to_cartesian((θ, ϕ))
    x = sin(θ)*cos(ϕ)
    y = sin(θ)*sin(ϕ) 
    z = cos(θ)
    return VectorValue(z, x, y)
end

export SpaceDimension, X, Y, Z
export SpaceDimensionality, _1D, _2D, _3D, dimensions, dimensionality_type
export cartesian_unit_vector, extend_3D, Ωx, Ωy, Ωz, to_Ω, from_Ω
export unitsphere_spherical_to_cartesian, unitsphere_cartesian_to_spherical

end