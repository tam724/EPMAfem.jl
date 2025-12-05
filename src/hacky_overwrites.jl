
if VERSION <= v"1.12"
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
end

function LinearAlgebra.kron!(z::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, α::Number, β::Number) where T
    @assert length(z) == length(x) * length(y)

    Z = reshape(z, (length(y), length(x)))
    Z .= α .* y .* transpose(x) .+ β .* Z

    return z
end

function LinearAlgebra.kron!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α::Number, β::Number)
    size(C) == LinearAlgebra._kronsize(A, B) || throw(DimensionMismatch("kron!"))
    m = firstindex(C)
    @inbounds for j in axes(A,2), l in axes(B,2), i in axes(A,1)
        Aij = A[i,j]
        for k in axes(B,1)
            C[m] = α*Aij*B[k,l] + β*C[m]
            m += 1
        end
    end
    return C
end

# https://github.com/JuliaLang/LinearAlgebra.jl/blob/b7fd6967c60b8408445d03442e04586bce0645d7/src/diagonal.jl#L809-L812
function LinearAlgebra.kron!(C::Diagonal, A::Diagonal, B::Diagonal)
    kron!(C.diag, A.diag, B.diag, true, false)
    return C
end

function LinearAlgebra.kron!(C::Diagonal, A::Diagonal, B::Diagonal, α::Number, β::Number)
    kron!(C.diag, A.diag, B.diag, α, β)
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

# we need transpose(A) * Diagonal() on CUDA:
function LinearAlgebra.mul!(B::CUDA.GPUArrays.AbstractGPUVecOrMat,
                            D::Diagonal{<:Any, <:CUDA.GPUArrays.AbstractGPUArray},
                            At::Transpose{<:Number, <:CUDA.GPUArrays.AbstractGPUVecOrMat},
                            α::Number,
                            β::Number)
    dd = D.diag
    d = length(dd)
    m, n = size(At, 1), size(At, 2)
    m′, n′ = size(B, 1), size(B, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    @. B = α * dd * At + β * B

    B
end

function LinearAlgebra.mul!(B::CUDA.GPUArrays.AbstractGPUVecOrMat,
                            At::Transpose{<:Number, <:CUDA.GPUArrays.AbstractGPUVecOrMat},
                            D::Diagonal{<:Any, <:CUDA.GPUArrays.AbstractGPUArray},
                            α::Number,
                            β::Number)
    dd = D.diag
    d = length(dd)
    m, n = size(At, 1), size(At, 2)
    m′, n′ = size(B, 1), size(B, 2)
    n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
    (m, n) == (m′, n′) || throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    ddT = transpose(dd)
    @. B = α * At * ddT + β * B

    B
end


## transpose! overloads
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/2c3fe9b7e0ca4e2c7bf506bd16ae5900f04a8023/src/transpose.jl#L41
function LinearAlgebra.transpose!(Bt::Transpose, A::AbstractMatrix)
    LinearAlgebra.check_transpose_axes(axes(Bt), axes(A))
    copyto!(parent(Bt), A)
    return Bt
end

function LinearAlgebra.transpose!(Bt::Transpose, A::AbstractMatrix, α::Number, β::Number)
    LinearAlgebra.check_transpose_axes(axes(Bt), axes(A))
    parent(Bt) .= α .* A .+ β .* parent(Bt)
    return Bt
end

LinearAlgebra.transpose!(B::AbstractMatrix, A::AbstractMatrix, α::Number, β::Number) = LinearAlgebra.transpose_f!(transpose, B, A, α, β)

# https://github.com/JuliaLang/LinearAlgebra.jl/blob/2c3fe9b7e0ca4e2c7bf506bd16ae5900f04a8023/src/transpose.jl#L99-L137
const transposebaselength=64
function LinearAlgebra.transpose_f!(f, B::AbstractMatrix, A::AbstractMatrix, α::Number, β::Number)
    inds = axes(A)
    LinearAlgebra.check_transpose_axes(axes(B), inds)

    m, n = length(inds[1]), length(inds[2])
    if m*n<=4*transposebaselength
        @inbounds begin
            for j = inds[2]
                for i = inds[1]
                    B[j, i] = α * f(A[i, j]) + β * B[j, i]
                end
            end
        end
    else
        LinearAlgebra.transposeblock!(f,B,A,m,n,first(inds[1])-1,first(inds[2])-1, α, β)
    end
    return B
end
function LinearAlgebra.transposeblock!(f, B::AbstractMatrix, A::AbstractMatrix, m::Int, n::Int, offseti::Int, offsetj::Int, α::Number, β::Number)
    if m * n<=transposebaselength
        @inbounds begin
            for j = offsetj .+ (1:n)
                for i = offseti .+ (1:m)
                    B[j, i] = α * f(A[i, j]) + β * B[j, i]
                end
            end
        end
    elseif m>n
        newm=m>>1
        LinearAlgebra.transposeblock!(f,B,A,newm,n,offseti,offsetj, α, β)
        LinearAlgebra.transposeblock!(f,B,A,m-newm,n,offseti+newm,offsetj, α, β)
    else
        newn=n>>1
        LinearAlgebra.transposeblock!(f,B,A,m,newn,offseti,offsetj, α, β)
        LinearAlgebra.transposeblock!(f,B,A,m,n-newn,offseti,offsetj+newn, α, β)
    end
    return B
end

# https://github.com/JuliaGPU/GPUArrays.jl/blob/49a339c63a50f1a00ac84844675bcb3a11070cb0/src/host/linalg.jl#L34C1-L44C4
@kernel function transpose_kernel_alphabeta!(B, @Const(A), alpha, beta)
    i, j = @index(Global, NTuple)
    @inbounds B[j, i] = alpha * A[i, j] + beta * B[j, i]
end

function LinearAlgebra.transpose!(B::GPUArrays.AbstractGPUArray, A::GPUArrays.AnyGPUMatrix, α::Number, β::Number)
    axes(B,1) == axes(A,2) && axes(B,2) == axes(A,1) || throw(DimensionMismatch("transpose"))
    transpose_kernel_alphabeta!(get_backend(B))(B, A, α, β; ndrange = size(A))
    return B
end

# empty triu!/tril!
function LinearAlgebra.tril!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
    if iszero(length(A)) return A end
    @kernel function tril_kernel!(_A, _d)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if i < j - _d
            @inbounds _A[i, j] = zero(T)
        end
    end
    tril_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
end

function LinearAlgebra.triu!(A::AbstractGPUMatrix{T}, d::Integer = 0) where T
    if iszero(length(A)) return A end
    @kernel function triu_kernel!(_A, _d)
        I = @index(Global, Cartesian)
        i, j = Tuple(I)
        if j < i + _d
            @inbounds _A[i, j] = zero(T)
        end
    end
    triu_kernel!(get_backend(A))(A, d; ndrange = size(A))
    return A
end

function (T::Type{<: AnyGPUArray{U}})(s::UniformScaling, dims::Dims{2}) where {U}
    res = similar(T, dims)
    fill!(res, zero(U))
    if iszero(minimum(dims)) return res end
    kernel = GPUArrays.identity_kernel(get_backend(res))
    kernel(res, size(res, 1), s.λ; ndrange=minimum(dims))
    return res
end

function LinearAlgebra.rmul!(A::CuArray{<:Union{Float32, Float64, ComplexF64, ComplexF32}}, val::Bool)
    if iszero(val) 
        return fill!(A, zero(eltype(A)))
    else
        return A
    end
end

function LinearAlgebra.rmul!(A::CuArray{<:Union{Float32, Float64, ComplexF64, ComplexF32}}, val::Int)
    if iszero(val)
        return fill!(A, zero(eltype(A)))
    else
        return rmul!(A, eltype(A)(val))
    end
end
