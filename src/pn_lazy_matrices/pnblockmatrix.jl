# now this is special code for EPMA
function blockmatrix(A, B, C, D)
    # weird hack: # TODO!
    if any(A -> A isa CUDA.CUSPARSE.CuSparseMatrixCSC, (A, B, C, D, transpose(A), transpose(B), transpose(C), transpose(D)))
        A_, B_, C_, D_ = collect.((A, B, C, D))
        return sparse([ A_ B_
                        C_ D_]) |> cu
    end
    return [A B
            C D]
end

const BlockMatrix{T} = LazyOpMatrix{T, typeof(blockmatrix), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}}}
A(BM::BlockMatrix) = BM.args[1]
B(BM::BlockMatrix) = BM.args[2]
C(BM::BlockMatrix) = BM.args[3]
D(BM::BlockMatrix) = BM.args[4]

A(BMt::Transpose{T, <:BlockMatrix{T}}) where T = transpose(A(parent(BMt)))
B(BMt::Transpose{T, <:BlockMatrix{T}}) where T = transpose(C(parent(BMt)))
C(BMt::Transpose{T, <:BlockMatrix{T}}) where T = transpose(B(parent(BMt)))
D(BMt::Transpose{T, <:BlockMatrix{T}}) where T = transpose(D(parent(BMt)))

block_size(BM::Union{BlockMatrix, Transpose{T, <:BlockMatrix{T}}}) where T = (
    only_unique((size(A(BM), 1), size(A(BM), 2), size(B(BM), 1), size(C(BM), 2))), 
    only_unique((size(D(BM), 1), size(D(BM), 2), size(B(BM), 2), size(C(BM), 1)))
)

# may be weaker
max_block_size(BM::Union{BlockMatrix, Transpose{T, <:BlockMatrix{T}}}) where T = (
    only_unique((max_size(A(BM), 1), max_size(A(BM), 2), max_size(B(BM), 1), max_size(C(BM), 2))), 
    only_unique((max_size(D(BM), 1), max_size(D(BM), 2), max_size(B(BM), 2), max_size(C(BM), 1)))
)

blocks(BM::Union{BlockMatrix, Transpose{T, <:BlockMatrix{T}}}) where T = A(BM), B(BM), C(BM), D(BM)

duplicate(x) = (x, x)
Base.size(BM::BlockMatrix) = duplicate(sum(block_size(BM)))
max_size(BM::Union{BlockMatrix, Transpose{T, <:BlockMatrix{T}}}) where T = duplicate(sum(max_block_size(BM)))
isdiagonal(BM::BlockMatrix) = false # would need B === 0

function lazy_getindex(BM::BlockMatrix, i::Int, j::Int)
    mA, nA = size(A(BM))

    if i <= mA && j <= nA
        return A(BM)[i, j]
    elseif i <= mA && j > nA
        return B(BM)[i, j - nA]
    elseif i > mA && j <= nA
        return C(BM)[i - mA, j]
    else
        return D(BM)[i - mA, j - nA]
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, BM::BlockMatrix, x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(BM)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_with!(ws, y1, A(BM), x1, α, β)
    mul_with!(ws, y1, B(BM), x2, α, true)

    mul_with!(ws, y2, C(BM), x1, α, β)
    mul_with!(ws, y2, D(BM), x2, α, true)
end

function mul_with!(ws::Workspace, y::AbstractVector, BMt::Transpose{T, <:BlockMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    n1, n2 = block_size(parent(BMt)) # is symmetric anyways...

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_with!(ws, y1, transpose(A(parent(BMt))), x1, α, β)
    mul_with!(ws, y1, transpose(C(parent(BMt))), x2, α, true)

    mul_with!(ws, y2, transpose(B(parent(BMt))), x1, α, β)
    mul_with!(ws, y2, transpose(D(parent(BMt))), x2, α, true)
end

required_workspace(::typeof(mul_with!), BM::BlockMatrix, cache_notifier) = maximum(required_workspace(mul_with!, A_, cache_notifier) for A_ in (A(BM), B(BM), C(BM), D(BM)))

materialize_with(ws::Workspace, BM::BlockMatrix, skeleton::AbstractMatrix) = materialize_with(ws, BM, skeleton, true, false)
function materialize_with(ws::Workspace, BM::BlockMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    n1, n2 = block_size(BM)
    materialize_with(ws, A(BM), @view(skeleton[1:n1, 1:n1]), α, β)
    materialize_with(ws, B(BM), @view(skeleton[1:n1, n1+1:n1+n2]), α, β)
    materialize_with(ws, C(BM), @view(skeleton[n1+1:n1+n2, 1:n1]), α, β)
    materialize_with(ws, D(BM), @view(skeleton[n1+1:n1+n2, n1+1:n1+n2]), α, β)
    return skeleton, ws
end
required_workspace(::typeof(materialize_with), BM::BlockMatrix, cache_notifier) = maximum(A -> required_workspace(materialize_with, A, cache_notifier), blocks(BM))

### INPLACE INV MATRIX
const InplaceInverseMatrix{T} = LazyOpMatrix{T, typeof(LinearAlgebra.inv!), <:Tuple{<:MaterializedMatrix{T}}}

@inline M(I::InplaceInverseMatrix) = only(I.args)
@inline A(I::InplaceInverseMatrix) = A(M(I))
Base.size(I::InplaceInverseMatrix) = size(A(I))
max_size(I::InplaceInverseMatrix) = max_size(A(I))
lazy_getindex(I::InplaceInverseMatrix, idx::Vararg{<:Integer}) = NaN
@inline isdiagonal(I::InplaceInverseMatrix) = isdiagonal(A(I))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, I::InplaceInverseMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    A_mat, _ = materialize_with(ws, M(I))
    @assert !β
    @assert α
    ldiv!(Y, A_mat, X)
end

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, It::Transpose{T, <:InplaceInverseMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    A_mat, _ = materialize_with(ws, M(parent(It)))
    @assert !β
    @assert α
    ldiv!(Y, transpose(A_mat), X)
end

required_workspace(::typeof(mul_with!), I::InplaceInverseMatrix, cache_notifier) = required_workspace(materialize_with, M(I), cache_notifier)

function materialize_with(ws::Workspace, I::InplaceInverseMatrix, skeleton::AbstractMatrix)
    CUDA.NVTX.@range "materialize inv" begin
        A_mat, _ = materialize_with(ws, A(I), skeleton)
    end
    CUDA.NVTX.@range "invert inv" begin
        LinearAlgebra.inv!(A_mat)
    end
    return A_mat, ws
end

required_workspace(::typeof(materialize_with), I::InplaceInverseMatrix, cache_notifier) = required_workspace(materialize_with, A(I), cache_notifier)
