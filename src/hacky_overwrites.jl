
# should come in julia 1.12: copied from https://github.com/JuliaLang/LinearAlgebra.jl/blob/b7fd6967c60b8408445d03442e04586bce0645d7/src/adjtrans.jl#L404-L413
 # these make eachrow(A') produce simpler views
@inline Base.unsafe_view(A::Transpose{<:Number, <:AbstractMatrix}, i::Integer, j::AbstractArray) =
    Base.unsafe_view(parent(A), j, i)
@inline Base.unsafe_view(A::Transpose{<:Number, <:AbstractMatrix}, i::AbstractArray, j::Integer) =
    Base.unsafe_view(parent(A), j, i)

@inline Base.unsafe_view(A::Adjoint{<:Real, <:AbstractMatrix}, i::Integer, j::AbstractArray) =
    Base.unsafe_view(parent(A), j, i)
@inline Base.unsafe_view(A::Adjoint{<:Real, <:AbstractMatrix}, i::AbstractArray, j::Integer) =
    Base.unsafe_view(parent(A), j, i)


# https://github.com/JuliaLang/LinearAlgebra.jl/blob/b7fd6967c60b8408445d03442e04586bce0645d7/src/diagonal.jl#L809-L812
function LinearAlgebra.kron!(C::Diagonal, A::Diagonal, B::Diagonal)
    kron!(C.diag, A.diag, B.diag)
    return C
end


# inplace inv! for Diagonal matrices
function LinearAlgebra.inv!(D::Diagonal{T})  where T
    D.diag .= inv.(D.diag)
    return D
end


# overload for the constructor of CuSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

# overload for the constructor of CuSparseVector that additionally converts the internal types.
function CUSPARSE.CuSparseVector{T, Ti}(v::SparseVector) where {T, Ti}
    return CUSPARSE.CuSparseVector{T, Ti}(
        CuVector{Ti}(v.nzind), CuVector{T}(v.nzval), length(v)
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

# A = x*transpose(y) for sparse vectors x and y and preallocated A
function LinearAlgebra.mul!(A::CUDA.CuMatrix, x::CUDA.CUSPARSE.CuSparseVector, y::Transpose{<:Any, <:CUDA.CuVector}, α::Number, β::Number)
    @assert size(A) == (length(x), length(y.parent))
    my_rmul!(A, β)
    x_nnz = SparseArrays.nnz(x)
    y_nnz = length(y.parent)
    if x_nnz <= 0 return A end
    @kernel function mul!_kernel(A, x_nz, x_i, y, α)
        inz, jnz = @index(Global, NTuple)
        A[x_i[inz], jnz] += α * x_nz[inz] * y[jnz]
    end
    backend = KernelAbstractions.get_backend(A)
    kernel! = mul!_kernel(backend)
    kernel!(A, SparseArrays.nonzeros(x), SparseArrays.nonzeroinds(x), y.parent, α, ndrange=(x_nnz, y_nnz))
    return A
end

# A = x*transpose(y) for sparse vectors x and y and preallocated A
function LinearAlgebra.mul!(A::CUDA.CuMatrix, x::CUDA.CuVector, y::Transpose{<:Any, <:CUDA.CUSPARSE.CuSparseVector}, α::Number, β::Number)
    @assert size(A) == (length(x), length(y.parent))
    my_rmul!(A, β)
    x_nnz = length(x)
    y_nnz = SparseArrays.nnz(y.parent)
    if y_nnz <= 0 return A end
    @kernel function mul!_kernel(A, x, y_nz, y_i, α)
        inz, jnz = @index(Global, NTuple)
        A[inz, y_i[jnz]] += α * x[inz] * y_nz[jnz]
    end
    backend = KernelAbstractions.get_backend(A)
    kernel! = mul!_kernel(backend)
    kernel!(A, x, SparseArrays.nonzeros(y.parent), SparseArrays.nonzeroinds(y.parent), α, ndrange=(x_nnz, y_nnz))
    return A
end

# fast check if SparseMatrix is diagonal
function LinearAlgebra.isdiag(A::SparseArrays.SparseMatrixCSC)
    return all(rv == cp for (rv, cp) ∈ zip(A.rowval, @view(A.colptr[1:end-1])))
end
