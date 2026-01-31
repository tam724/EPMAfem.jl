module BlockDiagonals
using LinearAlgebra
using SparseArrays
using Adapt

"""
    BlockDiagonal{T, N<:Integer, V<:AbstractVector{T}} <: AbstractMatrix{T}

Stores the values of a block-diagonal matrix using a single vector. The size of the blocks is NxN.
"""
struct BlockDiagonal{T, N, V<:AbstractVector{T}} <: AbstractMatrix{T}
    diag::V

    function BlockDiagonal{T, N, V}(diag) where {T, N, V<:AbstractVector{T}}
        Base.require_one_based_indexing(diag)
        N isa Integer && N >= 2 || error("N $N must be Integer and >= 2")
        rem(length(diag), N*N) == 0 || error("length(diag)=$(length(diag)) must be divisible by N^2=$(N*N).") 
        new{T, N, V}(diag)
    end
end

function BlockDiagonal{N}(diag::AbstractVector{T}) where {N, T}
    return BlockDiagonal{T, N, typeof(diag)}(diag)
end

function BlockDiagonal{1}(diag::AbstractVector)
    return Diagonal(diag)
end

function BlockDiagonal{N}(A::SparseMatrixCSC{T}) where {N, T}
    @assert size(A, 1) == size(A, 2)
    m = size(A, 1)
    @assert mod(m, N) == 0

    n_blocks = m ÷ N
    diag = zeros(T, n_blocks*N*N)

    # single pass over all nonzeros
    for j in 1:m
        bj = (j - 1) ÷ N
        col_offset = bj * N * N
        jj = (j - 1) % N

        for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[ptr]
            bi = (i - 1) ÷ N
            if bi != bj
                error("SparseMatrixCSC is not block-diagonal: nonzero at ($i, $j) lies outside diagonal blocks of size $N.")
            end

            ii = (i - 1) % N
            diag[col_offset + ii + jj * N + 1] = A.nzval[ptr]
        end
    end

    return BlockDiagonal{N}(diag)
end

n_blocks(A::BlockDiagonal{T, N}) where {T, N} = length(A.diag) ÷ (N*N)
duplicate(x) = (x, x)
Base.size(A::BlockDiagonal{T, N}) where {T, N} = duplicate(length(A.diag) ÷ N)
function Base.size(A::BlockDiagonal{T, N}, i) where {T, N}
    i > 0 || error("arraysize: dimension out of range")
    return i <= 2 ? length(A.diag) ÷ N : 1
end

function Base.getindex(A::BlockDiagonal{T, N}, i, j) where {T, N}
    # bounds check
    m = size(A, 1)
    (1 ≤ i ≤ m && 1 ≤ j ≤ m) || throw(BoundsError(A, (i, j)))

    # which block are we in?
    bi = (i - 1) ÷ N
    bj = (j - 1) ÷ N

    # off-diagonal block → zero
    if bi != bj
        return zero(T)
    end

    # local indices inside the block
    ii = (i - 1) % N
    jj = (j - 1) % N

    # index into diag (column-major)
    block_offset = bi * N * N
    return A.diag[block_offset + ii + jj * N + 1]
end

function Base.setindex!(A::BlockDiagonal{T, N}, v, i, j) where {T, N}
    # bounds check
    m = size(A, 1)
    (1 ≤ i ≤ m && 1 ≤ j ≤ m) || throw(BoundsError(A, (i, j)))

    # which block are we in?
    bi = (i - 1) ÷ N
    bj = (j - 1) ÷ N

    # off-diagonal block → zero
    if bi != bj
        iszero(v) || error("Cannot set off diagonal block element to nonzero")
        return 
    end

    # local indices inside the block
    ii = (i - 1) % N
    jj = (j - 1) % N

    # index into diag (column-major)
    block_offset = bi * N * N
    return A.diag[block_offset + ii + jj * N + 1] = v
end

Adapt.adapt_structure(to, x::BlockDiagonal{T, N}) where {T, N} = BlockDiagonal{N}(Adapt.adapt_structure(to, x.diag))

function LinearAlgebra.mul!(c::AbstractVector, A::BlockDiagonal{T, N}, b::AbstractVector, α::Number, β::Number) where {T, N}
    for i in 1:n_blocks(A)
        c_i = @view(c[(i-1)*N + 1: i*N])
        A_i = reshape(@view(A.diag[(i-1)*N*N + 1: i*N*N]), (N, N))
        b_i = @view(b[(i-1)*N + 1: i*N])
        mul!(c_i, A_i, b_i, α, β)
    end
    return c
end

function LinearAlgebra.mul!(C::AbstractMatrix, A::BlockDiagonal{T, N}, B::AbstractMatrix, α::Number, β::Number) where {T, N}
    for i in 1:n_blocks(A)
        C_i = @view(C[(i-1)*N + 1: i*N, :])
        A_i = reshape(@view(A.diag[(i-1)*N*N + 1: i*N*N]), (N, N))
        B_i = @view(B[(i-1)*N + 1: i*N, :])
        mul!(C_i, A_i, B_i, α, β)
    end
    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, B::AbstractMatrix, A::BlockDiagonal{T, N}, α::Number, β::Number) where {T, N}
    for i in 1:n_blocks(A)
        C_i = @view(C[:, (i-1)*N + 1: i*N])
        B_i = @view(B[:, (i-1)*N + 1: i*N])
        A_i = reshape(@view(A.diag[(i-1)*N*N + 1: i*N*N]), (N, N))
        mul!(C_i, B_i, A_i, α, β)
    end
    return C
end

function LinearAlgebra.inv!(A::BlockDiagonal{T, N}) where {T, N}
    for i in 1:n_blocks(A)
        A_i = reshape(@view(A.diag[(i-1)*N*N + 1: i*N*N]), (N, N))
        A_i .= inv(A_i)
    end
end


export BlockDiagonal

end
