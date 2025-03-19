# overload for the constructor of CUSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

function LinearAlgebra.isdiag(A::SparseArrays.SparseMatrixCSC)
    return all(rv == cp for (rv, cp) ∈ zip(A.rowval, @view(A.colptr[1:end-1])))
end

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
