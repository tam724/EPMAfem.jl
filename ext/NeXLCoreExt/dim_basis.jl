# This code is adapted from https://github.com/martinkosch/Dimensionless.jl (bust mostly stripped to the two core methods dimless/dimful)
# Maybe at some point it can be merged back into the original library 

"""
    DimBasis(basis_vectors...) -> DimBasis

Create a dimensional basis for a number of `basis_vectors` (quantities, units or dimensions).
"""
struct DimBasis{BV} end

function DimBasis(basis_vectors::SVector{N, <:Quantity}) where {N}
    # sanity check
    check_basis(dim_matrix(unique_dims(basis_vectors), basis_vectors))
    return DimBasis{basis_vectors.data}()
end

function DimBasis(basis_vectors::Vararg{<:Quantity, N}) where N
    return DimBasis(SVector{N}(basis_vectors))
end

basis_vectors(::DimBasis{BV}) where BV = SVector(BV)
basis_vectors(::Type{<:DimBasis{BV}}) where BV = SVector(BV)

basis_dimensions(basis) = unique_dims(basis_vectors(basis))


# Do not broadcast DimBasis
Base.Broadcast.broadcastable(basis::DimBasis) = Ref(basis)

dim_tuple(::Unitful.Dimensions{N}) where N = N
dim_tuple_types(::Unitful.Dimensions{N}) where N = typeof.(N)

Base.@pure Base.@inline unique_append(t) = t
Base.@pure Base.@inline unique_append(t, e, r...) = unique_append((e in t) ? t : (t..., e), r...)

Base.@pure Base.@inline unique_flatten(t, e) = unique_append(t, e...)
Base.@pure Base.@inline unique_flatten(t, e, r...) = unique_flatten(unique_append(t, e...), r...)
"""
    unique_dims(all_values...)

Return a vector of unique dimensions for `all_values`, a set of quantities, units or dimensions.
"""
function unique_dims(all_values::SVector{N, <:Unitful.Dimensions}) where N
    return SVector{N}(unique_flatten((), dim_tuple_types.(all_values)...))
end

unique_dims(all_values::SVector{N, <:Quantity}) where N = unique_dims(dimension.(all_values))

function dim_power(basis_dim, value::Unitful.Dimensions)
    dim_power = Rational(0)
    for dim in dim_tuple(value)
        if dim isa basis_dim
            dim_power = dim.power
        end
    end
    return dim_power
end

"""
    dim_matrix(basis_dims, all_values...)

Return the dimensional matrix for a set of basis dimensions `basis_dims` and `all_values`, a set of quantities, units or dimensions.
"""
function dim_matrix(basis_dims::SVector{NB}, all_values::SVector{NV, <:Unitful.Dimensions}) where {NB, NV}
    return SMatrix{NB, NV}([dim_power(basis_dim, value) for basis_dim in basis_dims, value in all_values])
end

function dim_vector(basis_dims::SVector{NB}, value::Unitful.Dimensions) where {NB}
    return SVector{NB}([dim_power(basis_dim, value) for basis_dim in basis_dims])
end

dim_vector(basis_dims, value::Quantity) = dim_vector(basis_dims, dimension(value))
dim_matrix(basis_dims, all_values::SVector{N, <:Quantity}) where N = dim_matrix(basis_dims, dimension.(all_values))

"""
    check_basis(dim_mat)

Use a dimensional matrix to check if a collection of dimensional vectors is a valid basis.
Throw errors if there are to few basis vectors or if the matrix does not have full rank.
"""
function check_basis(dim_mat)
    if size(dim_mat, 2) < size(dim_mat, 1)
        plr_sgl = (size(dim_mat, 2) == 1) ? "vector" : "vectors"
        error("Invalid basis: There are $(size(dim_mat, 1)) dimensions but only $(size(dim_mat, 2)) basis $(plr_sgl).")
    end

    mat_rank = LinearAlgebra.rank(dim_mat)
    if mat_rank < size(dim_mat, 2)
        if mat_rank == 1
            error("Invalid basis: There are $(size(dim_mat, 2)) basis vectors that are all linearly dependent.")
        else
            error("Invalid basis: There are $(size(dim_mat, 2)) basis vectors of which only $(mat_rank) are linearly independent.")
        end
    end
    nothing
end

"""
    fac_dimful(unit, basis)

Return the scalar, dimensionless factor that a dimensionless value has to be multiplied with in order to translate it into the given `unit` in the specified `basis`. 
"""
function fac_dimful_compute(unit::Unitful.Units, basis::DimBasis)
    dim_vec = dim_vector(basis_dimensions(basis), dimension(unit))
    dim_mat = dim_matrix(basis_dimensions(basis), basis_vectors(basis))
    basis_vec = basis_vectors(basis)
    fac = prod(basis_vec .^ (dim_mat \ dim_vec))
    return ustrip(uconvert(unit, fac))
end

@generated function fac_dimful(unit::Unitful.Units{N, D, A}, basis::DimBasis) where {N, D, A}
    fac = fac_dimful_compute(unit(), basis())
    return :(return $fac)
end


"""
    dimless(quantity, basis)

Make a `quantity` dimensionless using a dimensional `basis`.
"""
function dimless(quantity::Unitful.AbstractQuantity, basis::DimBasis)
    fac = fac_dimful(unit(quantity), basis)
    return ustrip(quantity) / fac
end

"""
    dimless(quantity_array, basis)

Make an `quantity_array` dimensionless using a dimensional `basis`.
"""
function dimless(quantity_array::AbstractArray{<:Unitful.AbstractQuantity}, basis::DimBasis)
    fac = fac_dimful(unit(eltype(quantity_array)), basis)
    return ustrip.(quantity_array) ./ fac
end


"""
    dimful(value, unit, basis)

Restore the `unit`s of a dimensionless `value` using a dimensional `basis`.
"""
function dimful(value, unit::Unitful.Units, basis::DimBasis)
    fac = fac_dimful(unit, basis)
    return value * fac * unit
end
