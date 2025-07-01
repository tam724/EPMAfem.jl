# now this is special code for EPMA
function blockmatrix(A, B, C)
    # weird hack: # TODO!
    if any(A -> A isa CUDA.CUSPARSE.CuSparseMatrixCSC, (A, B, C, transpose(A), transpose(B), transpose(C)))
        B_ = collect(B)
        return sparse([collect(A) B_
            transpose(B_) collect(C)]) |> cu
    end
    return [A               B
            transpose(B)    C]
end

const BlockMatrix{T} = LazyOpMatrix{T, typeof(blockmatrix), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}}}
A(BM::BlockMatrix) = BM.args[1]
B(BM::BlockMatrix) = BM.args[2]
C(BM::BlockMatrix) = BM.args[3]

block_size(BM::BlockMatrix) = (
    only_unique((size(A(BM), 1), size(A(BM), 2), size(B(BM), 1))), 
    only_unique((size(C(BM), 1), size(C(BM), 2), size(B(BM), 2)))
)

# may be weaker
max_block_size(BM::BlockMatrix) = (
    only_unique((max_size(A(BM), 1), max_size(A(BM), 2), max_size(B(BM), 1))), 
    only_unique((max_size(C(BM), 1), max_size(C(BM), 2), max_size(B(BM), 2)))
)

duplicate(x) = (x, x)
Base.size(BM::BlockMatrix) = duplicate(sum(block_size(BM)))
max_size(BM::BlockMatrix) = duplicate(sum(max_block_size(BM)))
isdiagonal(BM::BlockMatrix) = false # would need B === 0

function lazy_getindex(BM::BlockMatrix, i::Int, j::Int)
    mA, nA = size(A(BM))

    if i <= mA && j <= nA
        return A(BM)[i, j]
    elseif i <= mA && j > nA
        return B(BM)[i, j - nA]
    elseif i > mA && j <= nA
        return B(BM)[j, i - mA] # transpose(B)
    else
        return C(BM)[i - mA, j - nA]
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

    mul_with!(ws, y2, transpose(B(BM)), x1, α, β)
    mul_with!(ws, y2, C(BM), x2, α, true)
end

function mul_with!(ws::Workspace, y::AbstractVector, BMt::Transpose{T, <:BlockMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    n1, n2 = block_size(parent(BMt)) # is symmetric anyways...

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_with!(ws, y1, transpose(A(parent(BMt))), x1, α, β)
    mul_with!(ws, y1, B(parent(BMt)), x2, α, true)

    mul_with!(ws, y2, transpose(B(parent(BMt))), x1, α, β)
    mul_with!(ws, y2, transpose(C(parent(BMt))), x2, α, true)
end

required_workspace(::typeof(mul_with!), BM::BlockMatrix) = maximum(required_workspace(mul_with!, A_) for A_ in (A(BM), B(BM), C(BM)))

function materialize_with(ws::Workspace, BM::BlockMatrix)
    error("not implemented...")
end
required_workspace(::typeof(materialize_with), BM::BlockMatrix) = Inf

### INPLACE INV MATRIX
const InplaceInverseMatrix{T} = LazyOpMatrix{T, typeof(LinearAlgebra.inv!), <:Tuple{<:AbstractMatrix}}

@inline A(I::InplaceInverseMatrix) = first(I.args)
Base.size(I::InplaceInverseMatrix) = size(A(I))
max_size(I::InplaceInverseMatrix) = max_size(A(I))
lazy_getindex(I::InplaceInverseMatrix, idx::Vararg{<:Integer}) = undef
@inline isdiagonal(I::InplaceInverseMatrix) = isdiagonal(A(I))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, I::InplaceInverseMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    A_mat, rem = materialize_with(ws, materialize(A(I)), nothing)
    I_mat = LinearAlgebra.inv!(A_mat)
    mul!(Y, I_mat, X, α, β)
end

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, It::Transpose{T, <:InplaceInverseMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T
    At_mat, rem = materialize_with(ws, materialize(A(parent(It))), nothing)
    It_mat = LinearAlgebra.inv!(At_mat)
    mul!(Y, It_mat, X, α, β)
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, I::InplaceInverseMatrix, α::Number, β::Number)
    A_mat, rem = materialize_with(ws, materialize(A(I)), nothing)
    I_mat = LinearAlgebra.inv!(A_mat)
    mul!(Y, X, I_mat, α, β)
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, It::Transpose{T, <:InplaceInverseMatrix{T}}, α::Number, β::Number) where T
    At_mat, rem = materialize_with(ws, materialize(A(parent(It))), nothing)
    It_mat = LinearAlgebra.inv!(At_mat)
    mul!(Y, X, It_mat, α, β)
end

required_workspace(::typeof(mul_with!), I::InplaceInverseMatrix) = required_workspace(materialize_with, materialize(A(I)))

function materialize_with(ws::Workspace, I::InplaceInverseMatrix, ::Nothing)
    A_mat, rem = materialize_with(ws, materialize(A(I)), nothing)
    A_mat = inv!(A_mat)
    return A_mat, rem
end

function materialize_with(ws::Workspace, S::InplaceInverseMatrix, skeleton::AbstractMatrix)
    A_mat, _ = materialize_with(ws, materialize(A(S)), skeleton)
    skeleton .= a(S) .* A_mat
    return skeleton, ws
end

required_workspace(::typeof(materialize_with), I::InplaceInverseMatrix) = required_workspace(materialize_with, A(S))

function schur_complement(BM::BlockMatrix)
    C⁻¹ = lazy(LinearAlgebra.inv!, C(BM))
    BC⁻¹Bt = lazy(*, B(BM), C⁻¹, transpose(B(BM)))
    mBC⁻¹Bt = lazy(*, -one(eltype(BM)), BC⁻¹Bt)
    return lazy(+, A(BM), mBC⁻¹Bt)
end
