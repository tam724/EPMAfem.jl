# definitions for space dimensions
const Z = Val{1}
const X = Val{2}
const Y = Val{3}
dimension(D::Int) = (D in (1, 2, 3)) ? Val(D) : error("dimension D = $(D) must be 1, 2 or 3 (x, y or z)")

const _1D = Val{1}
const _2D = Val{2}
const _3D = Val{3}

dimensions(::_1D) = (Z(), )
dimensions(::_2D) = (Z(), X())
dimensions(::_3D) = (Z(), X(), Y())
dimensions() = (Z(), X(), Y())
dimensions(ND::Int) = (ND in (1, 2, 3)) ? dimensions(Val(ND)) : error("number of dimensions ND=$(ND) must be 1, 2 or 3 (x, y or z)")

cartesian_unit_vector(::Z, ::_1D) = VectorValue(1.0)

cartesian_unit_vector(::Z, ::_2D) = VectorValue(1.0, 0.0)
cartesian_unit_vector(::X, ::_2D) = VectorValue(0.0, 1.0)

cartesian_unit_vector(::Z, ::_3D) = VectorValue(1.0, 0.0, 0.0)
cartesian_unit_vector(::X, ::_3D) = VectorValue(0.0, 1.0, 0.0)
cartesian_unit_vector(::Y, ::_3D) = VectorValue(0.0, 0.0, 1.0)

extend_3D(x::VectorValue{1}) = VectorValue(x[1], 0.0, 0.0)
extend_3D(x::VectorValue{2}) = VectorValue(x[1], x[2], 0.0)
extend_3D(x::VectorValue{3}) = x

# see neming convention of https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
function unitsphere_cartesian_to_spherical(Ω::VectorValue{3})
    z, x, y = Ω

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