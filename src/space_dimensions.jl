module Dimensions

using Gridap: VectorValue

# definitions for space dimensions
abstract type SpaceDimension end
struct Z <: SpaceDimension end
struct X <: SpaceDimension end
struct Y <: SpaceDimension end
# dimension(D::Int) = (D in (1, 2, 3)) ? Val(D) : error("dimension D = $(D) must be 1, 2 or 3 (x, y or z)")

abstract type SpaceDimensionality end
struct _1D <: SpaceDimensionality end
struct _2D <: SpaceDimensionality end
struct _3D <: SpaceDimensionality end

abstract type SpaceBoundary end
struct LeftBoundary <: SpaceBoundary end
struct RightBoundary <: SpaceBoundary end

dimensions(::_1D) = (Z(), )
dimensions(::_2D) = (Z(), X())
dimensions(::_3D) = (Z(), X(), Y())
dimensions() = (Z(), X(), Y())

dimensionality(dim::SpaceDimensionality) = dim
function dimensionality(dim_int::Integer)
    if dim_int == 1
        return _1D()
    elseif dim_int == 2
        return _2D()
    elseif dim_int == 3
        return _3D()
    else
        error("number of dimensions dim_int=$(dim_int) must be 1, 2 or 3 (1D, 2D or 3D)")
    end
end

dimensionality_int(::_1D) = 1
dimensionality_int(::_2D) = 2
dimensionality_int(::_3D) = 3
dimensionality_int(ND::Integer) = ND ∈ (1, 2, 3) ? ND : error("number of dimensions ND=$(ND) must be 1, 2 or 3 (1D, 2D or 3D)")

cartesian_unit_vector(::Z, ::_1D) = VectorValue(1.0)

cartesian_unit_vector(::Z, ::_2D) = VectorValue(1.0, 0.0)
cartesian_unit_vector(::X, ::_2D) = VectorValue(0.0, 1.0)

cartesian_unit_vector(::Z, ::_3D) = VectorValue(1.0, 0.0, 0.0)
cartesian_unit_vector(::X, ::_3D) = VectorValue(0.0, 1.0, 0.0)
cartesian_unit_vector(::Y, ::_3D) = VectorValue(0.0, 0.0, 1.0)

outwards_normal(dim::SpaceDimension, ::RightBoundary, dims::SpaceDimensionality) = cartesian_unit_vector(dim, dims)
outwards_normal(dim::SpaceDimension, ::LeftBoundary, dims::SpaceDimensionality) = -cartesian_unit_vector(dim, dims)

select(x::VectorValue{3}, ::Z) = x[1]
select(x::VectorValue{3}, ::X) = x[2]
select(x::VectorValue{3}, ::Y) = x[3]

select(x::VectorValue{2}, ::Z) = x[1]
select(x::VectorValue{2}, ::X) = x[2]

select(x::VectorValue{1}, ::Z) = x[1]

Ωz(Ω::VectorValue{3}) = select(Ω, Z())
Ωx(Ω::VectorValue{3}) = select(Ω, X())
Ωy(Ω::VectorValue{3}) = select(Ω, Y())

Ωz(Ω::VectorValue{2}) = select(Ω, Z())
Ωx(Ω::VectorValue{2}) = select(Ω, X())

Ωz(Ω::VectorValue{1}) = select(Ω, Z())

omit(x::VectorValue{3}, ::Z) = (; x=select(x, X()), y=select(x, Y()))
omit(x::VectorValue{3}, ::X) = (; z=select(x, Z()), y=select(x, Y()))
omit(x::VectorValue{3}, ::Y) = (; z=select(x, Z()), x=select(x, X()))

omit(x::VectorValue{2}, ::Z) = (; x=select(x, X()))
omit(x::VectorValue{2}, ::X) = (; z=select(x, Z()))

omit(::VectorValue{1}, ::Z) = (; )

extend_3D(x::VectorValue{1}) = VectorValue(select(x, Z()), 0.0, 0.0)
extend_3D(x::VectorValue{2}) = VectorValue(select(x, Z()), select(x, X()), 0.0)
extend_3D(x::VectorValue{3}) = x

constrain(x::VectorValue{3}, ::_1D) = VectorValue(select(x, Z()))
constrain(x::VectorValue{3}, ::_2D) = VectorValue(select(x, Z()), select(x, X()))
constrain(x::VectorValue{3}, ::_3D) = x

to_args(x::VectorValue{3}) = (; z=select(x, Z()), x=select(x, X()), y=select(x, Y()))
to_args(x::VectorValue{2}) = (; z=select(x, Z()), x=select(x, X()))
to_args(x::VectorValue{1}) = (; z=select(x, Z()))

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
export SpaceDimensionality, _1D, _2D, _3D, dimensions, dimensionality
export cartesian_unit_vector, extend_3D, Ωx, Ωy, Ωz, to_Ω, from_Ω
export unitsphere_spherical_to_cartesian, unitsphere_cartesian_to_spherical

end
