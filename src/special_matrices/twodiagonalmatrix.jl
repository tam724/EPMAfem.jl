struct TwoDiagonalMatrix{T} <: AbstractMatrix{T}
    a::T
    b::T
    n::Int
end
Base.size(A::TwoDiagonalMatrix) = (A.n - 1, A.n)
function Base.getindex(A::TwoDiagonalMatrix{T}, i::Integer, j::Integer) where T
    if i == j
        return A.a
    elseif i + 1 == j
        return A.b
    else
        return zero(T)
    end
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::TwoDiagonalMatrix{T}, X::AbstractMatrix{T}, α::Number, β::Number) where T
    @assert size(Y, 1) + 1 == size(X, 1) == A.n
    @assert size(Y, 2) == size(X, 2)

    @kernel function AX_mul_kernel!(Y, αa, αb, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j] = Base.muladd(αa, X[i, j], αb * X[i+1, j])
    end

    @kernel function AX_mulβ_kernel!(Y, αa, αb, β, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j] = Base.muladd(αa, X[i, j], Base.muladd(αb, X[i+1, j], β * Y[i, j]))
    end

    αa = T(α * A.a)
    αb = T(α * A.b)

    backend = get_backend(Y)
    if iszero(β)
        kernel! = AX_mul_kernel!(backend)
        kernel!(Y, αa, αb, X, ndrange=size(Y))
    else
        kernel! = AX_mulβ_kernel!(backend)
        kernel!(Y, αa, αb, β, X, ndrange=size(Y))
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, A::TwoDiagonalMatrix{T}, α::Number, β::Number) where T
    @assert size(Y, 2) == size(X, 2) + 1 == A.n
    @assert size(Y, 1) == size(X, 1)

    @kernel function XA1_mul_kernel!(Y, αa, @Const(X))
        i = @index(Global)
        @inbounds Y[i, 1] = αa * X[i, 1]
    end

    @kernel function XA1_mulβ_kernel!(Y, αa, β, @Const(X))
        i = @index(Global)
        @inbounds Y[i, 1] = Base.muladd(αa, X[i, 1], β * Y[i, 1])
    end

    @kernel function XAn_mul_kernel!(Y, n, αb, @Const(X))
        i = @index(Global)
        @inbounds Y[i, n] = αb * X[i, n-1]
    end

    @kernel function XAn_mulβ_kernel!(Y, n, αb, β, @Const(X))
        i = @index(Global)
        @inbounds Y[i, n] = Base.muladd(αb, X[i, n-1], β * Y[i, n])
    end

    @kernel function XA_mul_kernel!(Y, αa, αb, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j+1] = Base.muladd(αa, X[i, j+1], αb * X[i, j])
    end

    @kernel function XA_mulβ_kernel!(Y, αa, αb, β, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j+1] = Base.muladd(αa, X[i, j+1], Base.muladd(αb, X[i, j], β * Y[i, j+1]))
    end

    αa = T(α * A.a)
    αb = T(α * A.b)
    
    backend = get_backend(Y)
    if iszero(β)
        kernel! = XA1_mul_kernel!(backend)
        kernel!(Y, αa, X, ndrange=(size(Y, 1)))
        kernel! = XA_mul_kernel!(backend)
        kernel!(Y, αa, αb, X, ndrange=(size(Y, 1), size(Y, 2)-2))
        kernel! = XAn_mul_kernel!(backend)
        kernel!(Y, A.n, αb, X, ndrange=(size(Y, 1)))
    else
        kernel! = XA1_mulβ_kernel!(backend)
        kernel!(Y, αa, β, X, ndrange=(size(Y, 1)))
        kernel! = XA_mulβ_kernel!(backend)
        kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1), size(Y, 2)-2))
        kernel! = XAn_mulβ_kernel!(backend)
        kernel!(Y, A.n, αb, β, X, ndrange=(size(Y, 1)))
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, At::Transpose{T,<:TwoDiagonalMatrix{T}}, X::AbstractMatrix{T}, α::Number, β::Number) where T
    n = parent(At).n
    @assert size(Y, 1) == size(X, 1) + 1 == n
    @assert size(Y, 2) == size(X, 2)

    # on CPU launching 3 kernels is faster than branching in kernel (on GPU same runtime)
    @kernel function AtX1_mul_kernel!(Y, αa, @Const(X))
        j = @index(Global)
        @inbounds Y[1, j] = αa * X[1, j]
    end

    @kernel function AtX1_mulβ_kernel!(Y, αa, β, @Const(X))
        j = @index(Global)
        @inbounds Y[1, j] = Base.muladd(αa, X[1, j], β * Y[1, j])
    end

    @kernel function AtXn_mul_kernel!(Y, n, αb, @Const(X))
        j = @index(Global)
        @inbounds Y[n, j] = αb * X[n-1, j]
    end

    @kernel function AtXn_mulβ_kernel!(Y, n, αb, β, @Const(X))
        j = @index(Global)
        @inbounds Y[n, j] = Base.muladd(αb, X[n-1, j], β * Y[n, j])
    end

    @kernel function AtX_mul_kernel!(Y, αa, αb, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i+1, j] = αa * X[i+1, j] + αb * X[i, j]
    end

    @kernel function AtX_mulβ_kernel!(Y, αa, αb, β, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i+1, j] = Base.muladd(αa, X[i+1, j], Base.muladd(αb, X[i, j], β * Y[i+1, j]))
    end

    αa = T(α * parent(At).a)
    αb = T(α * parent(At).b)

    backend = get_backend(Y)
    if iszero(β)
        kernel! = AtX1_mul_kernel!(backend)
        kernel!(Y, αa, X, ndrange=(size(Y, 2)))

        kernel! = AtX_mul_kernel!(backend)
        kernel!(Y, αa, αb, X, ndrange=(size(Y, 1) - 2, size(Y, 2)))

        kernel! = AtXn_mul_kernel!(backend)
        kernel!(Y, n, αb, X, ndrange=(size(Y, 2)))
    else
        kernel! = AtX1_mulβ_kernel!(backend)
        kernel!(Y, αa, β, X, ndrange=(size(Y, 2)))

        kernel! = AtX_mulβ_kernel!(backend)
        kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1) - 2, size(Y, 2)))

        kernel! = AtXn_mulβ_kernel!(backend)
        kernel!(Y, n, αb, β, X, ndrange=(size(Y, 2)))
    end

    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, At::Transpose{T, <:TwoDiagonalMatrix{T}}, α::Number, β::Number) where T
    n = parent(At).n
    @assert size(Y, 2) + 1 == size(X, 2) == n
    @assert size(Y, 1) == size(X, 1)

    @kernel function XAt_mul_kernel!(Y, αa, αb, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j] = Base.muladd(αa, X[i, j], αb * X[i, j+1])
    end

    @kernel function XAt_mulβ_kernel!(Y, αa, αb, β, @Const(X))
        i, j = @index(Global, NTuple)
        @inbounds Y[i, j] = Base.muladd(αa, X[i, j], Base.muladd(αb, X[i, j+1], β * Y[i, j]))
    end

    αa = T(α * parent(At).a)
    αb = T(α * parent(At).b)

    backend = get_backend(Y)
    if iszero(β)
        kernel! = XAt_mul_kernel!(backend)
        kernel!(Y, αa, αb, X, ndrange=size(Y))
    else
        kernel! = XAt_mulβ_kernel!(backend)
        kernel!(Y, αa, αb, β, X, ndrange=size(Y))
    end
    return Y
end

