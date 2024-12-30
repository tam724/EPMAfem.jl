module BlockedMatrices

using LinearAlgebra
using ConcreteStructs

@concrete struct BlockedMatrix{T} <: AbstractMatrix{T}
    blocks
    indices
    axes
end

function check_block(block, block_indices, axes)
    @assert size(block) == (length(block_indices[1]), length(block_indices[2]))
    @assert all(i -> i ∈ axes[1], block_indices[1])
    @assert all(i -> i ∈ axes[2], block_indices[2])
end

function BlockedMatrix(blocks, indices, axes)
    T = eltype(first(blocks))
    @assert all(b -> b isa AbstractMatrix{T}, blocks)
    for (b, idx) in zip(blocks, indices)
        check_block(b, idx, axes)
    end
    return BlockedMatrix{T}(blocks, indices, axes)
end

function blocked_from_mat(A, indices, warn=true)
    blocks = tuple((A[inds[1], inds[2]] for inds in indices)...)
    B = BlockedMatrix(blocks, indices, axes(A))
    !warn || A ≈ B || @warn "Blocked matrix is not equal to original matrix"
    return B
end

Base.size(A::BlockedMatrix) = length.(A.axes)
Base.size(A::BlockedMatrix, i) = length(A.axes[i])
Base.eltype(::BlockedMatrix{T}) where T = T
Base.:≈(A::AbstractMatrix, B::BlockedMatrix) = A ≈ collect(B)
Base.:≈(A::BlockedMatrix, B::BlockedMatrix) = collect(A) ≈ collect(B)
Base.:≈(B::BlockedMatrix, A::AbstractMatrix) = collect(B) ≈ A
num_blocks(A::BlockedMatrix) = length(A.blocks)

Base.show(io::IO, A::BlockedMatrix) = print(io, "$(size(A, 1))x$(size(A, 2)) BlockedMatrix{$(eltype(A))} with blocks: $(["$(length(A.indices[i][1]))x$(length(A.indices[i][2]))" * ((i!=num_blocks(A)) ? ", " : "") for i in 1:num_blocks(A)]...)")
Base.show(io::IO, m::MIME"text/plain", A::BlockedMatrix) = print(io, "$(size(A, 1))x$(size(A, 2)) BlockedMatrix{$(eltype(A))} with blocks: $(["$(length(A.indices[i][1]))x$(length(A.indices[i][2]))" * ((i!=num_blocks(A)) ? ", " : "") for i in 1:num_blocks(A)]...)")

function Base.getindex(A::BlockedMatrix{T}, i::Integer, j::Integer, warn=true) where T
    !warn || @warn "Accessing elements of BlockedMatrix is slow"
    for (b, idx) in zip(A.blocks, A.indices)
        if i ∈ idx[1] && j ∈ idx[2]
            i_b = i - idx[1].start+1
            j_b = j - idx[2].start+1
            return b[i_b, j_b]
        end
    end
    return zero(T)
end

function Base.collect(A::BlockedMatrix)
    A_full = zeros(size(A))
    for (inds, block) in zip(A.indices, A.blocks)
        A_full[inds[1], inds[2]] .= block
    end
    return A_full
end

function Base.collect(AT::Transpose{T, <:BlockedMatrix{T}}) where T
    AT_full = zeros(size(AT))
    A = AT.parent
    for (inds, block) in zip(A.indices, A.blocks)
        AT_full[inds[2], inds[1]] .= transpose(block)
    end
    return AT_full
end

function LinearAlgebra.mul!(C::AbstractMatrix, B::AbstractMatrix, A::BlockedMatrix, α::Number, β::Number)
    for i in eachindex(A.blocks)
        inds = A.indices[i]
        block = A.blocks[i]
        mul!(@view(C[:, inds[2]]), @view(B[:, inds[1]]), block, α, β)
    end
    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix, B::AbstractMatrix, AT::Transpose{T, <:BlockedMatrix{T}}, α::Number, β::Number) where T
    A = AT.parent
    for i in eachindex(A.blocks)
        inds = A.indices[i]
        block = A.blocks[i]
        mul!(@view(C[:, inds[1]]), @view(B[:, inds[2]]), transpose(block), α, β)
    end
    return C
end
end