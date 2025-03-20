# overload for the constructor of CUSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

# A = x*transpose(y) for sparse vectors x and y and preallocated A
function LinearAlgebra.mul!(A::CUDA.CuMatrix, x::CUDA.CUSPARSE.CuSparseVector, y::Transpose{<:Any, <:CUDA.CUSPARSE.CuSparseVector}, α::Number, β::Number)
    @assert size(A) == (length(x), length(y.parent))
    my_rmul!(A, β)
    x_nnz = SparseArrays.nnz(x)
    y_nnz = SparseArrays.nnz(y.parent)
    if x_nnz <= 0 || y_nnz <= 0 return A end
    @kernel function mul!_kernel(A, x_nz, x_i, y_nz, y_i, α)
        inz, jnz = @index(Global, NTuple)
        A[x_i[inz], y_i[jnz]] += α * x_nz[inz] * y_nz[jnz]
    end
    backend = KernelAbstractions.get_backend(A)
    kernel! = mul!_kernel(backend)
    kernel!(A, SparseArrays.nonzeros(x), SparseArrays.nonzeroinds(x), SparseArrays.nonzeros(y.parent), SparseArrays.nonzeroinds(y.parent), α, ndrange=(x_nnz, y_nnz))
    return A
end

# fast check if SparseMatrix is diagonal
function LinearAlgebra.isdiag(A::SparseArrays.SparseMatrixCSC)
    return all(rv == cp for (rv, cp) ∈ zip(A.rowval, @view(A.colptr[1:end-1])))
end

## convert to Diagonal type if A is diagonal
diag_if_diag(A::Diagonal) = A
function diag_if_diag(A::AbstractMatrix)
    if isdiag(A)
        return Diagonal(Vector(diag(A)))
    else
        return A
    end
end

function dot_buf(x::AbstractVector, A::AbstractMatrix, y::AbstractVector, buf::AbstractVector)
    mul!(transpose(buf), transpose(x), A)
    return dot(y, buf)
end
