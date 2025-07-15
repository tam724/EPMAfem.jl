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

required_workspace(::typeof(mul_with!), BM::BlockMatrix) = maximum(required_workspace(mul_with!, A_) for A_ in (A(BM), B(BM), C(BM), D(BM)))

function materialize_with(ws::Workspace, BM::BlockMatrix)
    error("not implemented...")
end
required_workspace(::typeof(materialize_with), BM::BlockMatrix) = Inf

### INPLACE INV MATRIX
const InplaceInverseMatrix{T} = LazyOpMatrix{T, typeof(LinearAlgebra.inv!), <:Tuple{<:AbstractMatrix}}

@inline A(I::InplaceInverseMatrix) = only(I.args)
Base.size(I::InplaceInverseMatrix) = size(A(I))
max_size(I::InplaceInverseMatrix) = max_size(A(I))
lazy_getindex(I::InplaceInverseMatrix, idx::Vararg{<:Integer}) = NaN
@inline isdiagonal(I::InplaceInverseMatrix) = isdiagonal(A(I))
LinearAlgebra.transpose(I::InplaceInverseMatrix) = lazy(LinearAlgebra.inv!, transpose(A(I)))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, I::InplaceInverseMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    A_mat, _ = materialize_with(ws, materialize(A(I)), nothing)
    @assert !β
    @assert α
    ldiv!(Y, A_mat, X)
end

required_workspace(::typeof(mul_with!), I::InplaceInverseMatrix) = required_workspace(materialize_with, materialize(A(I)))

function materialize_with(ws::Workspace, I::InplaceInverseMatrix, skeleton::AbstractMatrix)
    A_mat, _ = materialize_with(ws, A(I), skeleton)
    A_mat .= LinearAlgebra.inv!(A_mat)
    return A_mat, ws
end

required_workspace(::typeof(materialize_with), I::InplaceInverseMatrix) = required_workspace(materialize_with, A(I))
