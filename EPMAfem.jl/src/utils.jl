# overload for the constructor of CUSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

function to_diag(VT, A::Matrix)
    @assert isdiag(A)
    return Diagonal(VT(diag(A)))
end

function to_diag(VT, A::Diagonal)
    @assert isdiag(A)
    return Diagonal(VT(A.diag))
end
