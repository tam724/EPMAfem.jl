module RowSparseMatrices
    using ConcreteStructs
    using LinearAlgebra
    using SparseArrays

    @concrete struct RowSparseMatrix{T} <: AbstractMatrix{T}
        nonzero_rows
        m
        data
    end

    function RowSparseMatrix(nonzero_rows::AbstractVector{Ti}, m::Ti, data::AbstractMatrix{T}) where {T, Ti <: Integer}
        @assert issorted(nonzero_rows)
        @assert all(i -> i <= m, nonzero_rows)
        @assert length(nonzero_rows) == size(data, 1)
        return RowSparseMatrix{T}(nonzero_rows, m, data)
    end

    function findfirst_sorted(v, i)
        idx = searchsortedfirst(v, i)
        if idx <= length(v) && v[idx] == i
            return idx
        else
            return nothing
        end
    end

    Base.size(A::RowSparseMatrix) = (A.m, size(A.data, 2))
    function Base.getindex(A::RowSparseMatrix{T}, i, j) where T
        idx = findfirst_sorted(A.nonzero_rows, i)
        if isnothing(idx)
            return zero(T)
        else
            return A.data[idx, j]
        end
    end

    function LinearAlgebra.svd(A::RowSparseMatrix{T}; full=false, alg::LinearAlgebra.Algorithm=LinearAlgebra.default_svd_alg(A.data)) where T
        svd_ = svd(A.data; full=full, alg=alg)
        return LinearAlgebra.SVD{T, T, AbstractMatrix{T}, typeof(svd_.S)}(RowSparseMatrix(A.nonzero_rows, A.m, svd_.U), svd_.S, svd_.Vt)
    end

    function LinearAlgebra.svd(At::Transpose{T, <:RowSparseMatrix{T}}; full=false, alg::LinearAlgebra.Algorithm=LinearAlgebra.default_svd_alg(parent(At).data)) where T
        svd_ = svd(parent(At); full=full, alg=alg)
        return LinearAlgebra.SVD{T, T, AbstractMatrix{T}, typeof(svd_.S)}(transpose(svd_.Vt), svd_.S, transpose(svd_.U))
    end

    function LinearAlgebra.mul!(y::AbstractVector, A::RowSparseMatrix, x::AbstractVector, α::Number, β::Number)
        rmul!(y, β)
        mul!(@view(y[A.nonzero_rows]), A.data, x, α, one(β))
        return y
    end

    function LinearAlgebra.mul!(y::AbstractVector, At::Transpose{T, <:RowSparseMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
        A = parent(At)
        mul!(y, transpose(A.data), @view(x[A.nonzero_rows]), α, β)
        return y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, A::RowSparseMatrix, X::AbstractMatrix, α::Number, β::Number)
        rmul!(Y, β)
        mul!(@view(Y[A.nonzero_rows, :]), A.data, X, α, one(β))
        return Y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, At::Transpose{T, <:RowSparseMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
        A = parent(At)
        mul!(Y, transpose(A.data), @view(X[A.nonzero_rows, :]), α, β)
        return Y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::RowSparseMatrix, α::Number, β::Number)
        mul!(Y, @view(X[:, A.nonzero_rows]), A.data, α, β)
        return Y
    end

    function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, At::Transpose{T, <:RowSparseMatrix{T}}, α::Number, β::Number) where T
        A = parent(At)
        rmul!(Y, β)
        mul!(@view(Y[:, A.nonzero_rows]), X, transpose(A.data), α, one(β))
        return Y
    end

    export RowSparseMatrix
end
