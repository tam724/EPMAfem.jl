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

## AX
@kernel function AX_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[i, j, k], αb * X[i+1, j, k])
end

@kernel function AX_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[i, j, k], Base.muladd(αb, X[i+1, j, k], β * Y[i, j, k]))
end

# AXt
@kernel function AXt_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[j, i, k], αb * X[j, i+1, k])
end

@kernel function AXt_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[j, i, k], Base.muladd(αb, X[j, i+1, k], β * Y[i, j, k]))
end

function batched_mul!(Y::AbstractArray{T, 3}, A::TwoDiagonalMatrix{T}, X::AbstractArray{T, 3}, Xt::Bool, α::Number, β::Number) where T
    if Xt
        @assert size(Y, 1) + 1 == size(X, 2) == A.n
        @assert size(Y, 2) == size(X, 1)
        @assert size(Y, 3) == size(X, 3)
    else
        @assert size(Y, 1) + 1 == size(X, 1) == A.n
        @assert size(Y, 2) == size(X, 2)
        @assert size(Y, 3) == size(X, 3)
    end

    αa = T(α * A.a)
    αb = T(α * A.b)

    backend = get_backend(Y)
    if Xt
        if iszero(β)
            kernel! = AXt_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=size(Y))
        else
            kernel! = AXt_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=size(Y))
        end
    else
        if iszero(β)
            kernel! = AX_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=size(Y))
        else
            kernel! = AX_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=size(Y))
        end
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::TwoDiagonalMatrix{T}, X::AbstractMatrix{T}, α::Number, β::Number) where T
    batched_mul!(reshape(Y, (size(Y)..., 1)), A, reshape(X, (size(X)..., 1)), false, α, β)
    return Y
end

## XA
@kernel function XA1_mul_kernel!(Y, αa, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, 1, k] = αa * X[i, 1, k]
end

@kernel function XA1_mulβ_kernel!(Y, αa, β, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, 1, k] = Base.muladd(αa, X[i, 1, k], β * Y[i, 1, k])
end

@kernel function XAn_mul_kernel!(Y, n, αb, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, n, k] = αb * X[i, n-1, k]
end

@kernel function XAn_mulβ_kernel!(Y, n, αb, β, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, n, k] = Base.muladd(αb, X[i, n-1, k], β * Y[i, n, k])
end

@kernel function XA_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j+1, k] = Base.muladd(αa, X[i, j+1, k], αb * X[i, j, k])
end

@kernel function XA_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j+1, k] = Base.muladd(αa, X[i, j+1, k], Base.muladd(αb, X[i, j, k], β * Y[i, j+1, k]))
end

## XAt
@kernel function XtA1_mul_kernel!(Y, αa, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, 1, k] = αa * X[1, i, k]
end

@kernel function XtA1_mulβ_kernel!(Y, αa, β, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, 1, k] = Base.muladd(αa, X[1, i, k], β * Y[i, 1, k])
end

@kernel function XtAn_mul_kernel!(Y, n, αb, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, n, k] = αb * X[n-1, i, k]
end

@kernel function XtAn_mulβ_kernel!(Y, n, αb, β, @Const(X))
    i, k = @index(Global, NTuple)
    @inbounds Y[i, n, k] = Base.muladd(αb, X[n-1, i, k], β * Y[i, n, k])
end

@kernel function XtA_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j+1, k] = Base.muladd(αa, X[j+1, i, k], αb * X[j, i, k])
end

@kernel function XtA_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j+1, k] = Base.muladd(αa, X[j+1, i, k], Base.muladd(αb, X[j, i, k], β * Y[i, j+1, k]))
end

function batched_mul!(Y::AbstractArray{T}, X::AbstractArray{T}, A::TwoDiagonalMatrix{T}, Xt::Bool, α::Number, β::Number) where T
    @assert Xt == false
    if Xt
        @assert size(Y, 2) == size(X, 1) + 1 == A.n
        @assert size(Y, 1) == size(X, 2)
        @assert size(Y, 3) == size(X, 3)
    else
        @assert size(Y, 2) == size(X, 2) + 1 == A.n
        @assert size(Y, 1) == size(X, 1)
        @assert size(Y, 3) == size(X, 3)
    end

    αa = T(α * A.a)
    αb = T(α * A.b)
    
    backend = get_backend(Y)
    if Xt
        if iszero(β)
            kernel! = XtA1_mul_kernel!(backend)
            kernel!(Y, αa, X, ndrange=(size(Y, 1), size(Y, 3)))
            kernel! = XtA_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=(size(Y, 1), size(Y, 2)-2, size(Y, 3)))
            kernel! = XtAn_mul_kernel!(backend)
            kernel!(Y, A.n, αb, X, ndrange=(size(Y, 1), size(Y, 3)))
        else
            kernel! = XtA1_mulβ_kernel!(backend)
            kernel!(Y, αa, β, X, ndrange=(size(Y, 1), size(Y, 3)))
            kernel! = XtA_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1), size(Y, 2)-2, size(Y, 3)))
            kernel! = XtAn_mulβ_kernel!(backend)
            kernel!(Y, A.n, αb, β, X, ndrange=(size(Y, 1), size(Y, 3)))
        end
    else
        if iszero(β)
            kernel! = XA1_mul_kernel!(backend)
            kernel!(Y, αa, X, ndrange=(size(Y, 1), size(Y, 3)))
            kernel! = XA_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=(size(Y, 1), size(Y, 2)-2, size(Y, 3)))
            kernel! = XAn_mul_kernel!(backend)
            kernel!(Y, A.n, αb, X, ndrange=(size(Y, 1), size(Y, 3)))
        else
            kernel! = XA1_mulβ_kernel!(backend)
            kernel!(Y, αa, β, X, ndrange=(size(Y, 1), size(Y, 3)))
            kernel! = XA_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1), size(Y, 2)-2, size(Y, 3)))
            kernel! = XAn_mulβ_kernel!(backend)
            kernel!(Y, A.n, αb, β, X, ndrange=(size(Y, 1), size(Y, 3)))
        end
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, A::TwoDiagonalMatrix{T}, α::Number, β::Number) where T
    batched_mul!(reshape(Y, (size(Y)..., 1)), reshape(X, (size(X)..., 1)), A, false, α, β)
    return Y
end

# AtX
@kernel function AtX1_mul_kernel!(Y, αa, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[1, j, k] = αa * X[1, j, k]
end

@kernel function AtX1_mulβ_kernel!(Y, αa, β, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[1, j, k] = Base.muladd(αa, X[1, j, k], β * Y[1, j, k])
end

@kernel function AtXn_mul_kernel!(Y, n, αb, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[n, j, k] = αb * X[n-1, j, k]
end

@kernel function AtXn_mulβ_kernel!(Y, n, αb, β, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[n, j, k] = Base.muladd(αb, X[n-1, j, k], β * Y[n, j, k])
end

@kernel function AtX_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i+1, j, k] = αa * X[i+1, j, k] + αb * X[i, j, k]
end

@kernel function AtX_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i+1, j, k] = Base.muladd(αa, X[i+1, j, k], Base.muladd(αb, X[i, j, k], β * Y[i+1, j, k]))
end

# AtXt
@kernel function AtXt1_mul_kernel!(Y, αa, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[1, j, k] = αa * X[j, 1, k]
end

@kernel function AtXt1_mulβ_kernel!(Y, αa, β, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[1, j, k] = Base.muladd(αa, X[j, 1, k], β * Y[1, j, k])
end

@kernel function AtXtn_mul_kernel!(Y, n, αb, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[n, j, k] = αb * X[j, n-1, k]
end

@kernel function AtXtn_mulβ_kernel!(Y, n, αb, β, @Const(X))
    j, k = @index(Global, NTuple)
    @inbounds Y[n, j, k] = Base.muladd(αb, X[j, n-1, k], β * Y[n, j, k])
end

@kernel function AtXt_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i+1, j, k] = αa * X[j, i+1, k] + αb * X[j, i, k]
end

@kernel function AtXt_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i+1, j, k] = Base.muladd(αa, X[j, i+1, k], Base.muladd(αb, X[j, i, k], β * Y[i+1, j, k]))
end

function batched_mul!(Y::AbstractArray{T}, At::Transpose{T,<:TwoDiagonalMatrix{T}}, X::AbstractArray{T}, Xt::Bool, α::Number, β::Number) where T
    n = parent(At).n
    if Xt
        @assert size(Y, 1) == size(X, 2) + 1 == n
        @assert size(Y, 2) == size(X, 1)
        @assert size(Y, 3) == size(X, 3)
    else
        @assert size(Y, 1) == size(X, 1) + 1 == n
        @assert size(Y, 2) == size(X, 2)
        @assert size(Y, 3) == size(X, 3)
    end

    # on CPU launching 3 kernels is faster than branching in kernel (on GPU same runtime)
    αa = T(α * parent(At).a)
    αb = T(α * parent(At).b)

    backend = get_backend(Y)
    if Xt
        if iszero(β)
            kernel! = AtXt1_mul_kernel!(backend)
            kernel!(Y, αa, X, ndrange=(size(Y, 2), size(Y, 3)))

            kernel! = AtXt_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=(size(Y, 1) - 2, size(Y, 2), size(Y, 3)))

            kernel! = AtXtn_mul_kernel!(backend)
            kernel!(Y, n, αb, X, ndrange=(size(Y, 2), size(Y, 3)))
        else
            kernel! = AtXt1_mulβ_kernel!(backend)
            kernel!(Y, αa, β, X, ndrange=(size(Y, 2), size(Y, 3)))

            kernel! = AtXt_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1) - 2, size(Y, 2), size(Y, 3)))

            kernel! = AtXtn_mulβ_kernel!(backend)
            kernel!(Y, n, αb, β, X, ndrange=(size(Y, 2), size(Y, 3)))
        end
    else
        if iszero(β)
            kernel! = AtX1_mul_kernel!(backend)
            kernel!(Y, αa, X, ndrange=(size(Y, 2), size(Y, 3)))

            kernel! = AtX_mul_kernel!(backend)
            kernel!(Y, αa, αb, X, ndrange=(size(Y, 1) - 2, size(Y, 2), size(Y, 3)))

            kernel! = AtXn_mul_kernel!(backend)
            kernel!(Y, n, αb, X, ndrange=(size(Y, 2), size(Y, 3)))
        else
            kernel! = AtX1_mulβ_kernel!(backend)
            kernel!(Y, αa, β, X, ndrange=(size(Y, 2), size(Y, 3)))

            kernel! = AtX_mulβ_kernel!(backend)
            kernel!(Y, αa, αb, β, X, ndrange=(size(Y, 1) - 2, size(Y, 2), size(Y, 3)))

            kernel! = AtXn_mulβ_kernel!(backend)
            kernel!(Y, n, αb, β, X, ndrange=(size(Y, 2), size(Y, 3)))
        end
    end
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, At::Transpose{T,<:TwoDiagonalMatrix{T}}, X::AbstractMatrix{T}, α::Number, β::Number) where T
    batched_mul!(reshape(Y, (size(Y)..., 1)), At, reshape(X, (size(X)..., 1)), false, α, β)
    return Y
end

## XAt

@kernel function XAt_mul_kernel!(Y, αa, αb, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[i, j, k], αb * X[i, j+1, k])
end

@kernel function XAt_mulβ_kernel!(Y, αa, αb, β, @Const(X))
    i, j, k = @index(Global, NTuple)
    @inbounds Y[i, j, k] = Base.muladd(αa, X[i, j, k], Base.muladd(αb, X[i, j+1, k], β * Y[i, j, k]))
end

function batched_mul!(Y::AbstractArray{T}, X::AbstractArray{T}, At::Transpose{T, <:TwoDiagonalMatrix{T}}, Xt::Bool, α::Number, β::Number) where T
    @assert Xt == false
    n = parent(At).n
    @assert size(Y, 2) + 1 == size(X, 2) == n
    @assert size(Y, 1) == size(X, 1)
    @assert size(Y, 3) == size(X, 3)

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

function LinearAlgebra.mul!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, At::Transpose{T, <:TwoDiagonalMatrix{T}}, α::Number, β::Number) where T
    batched_mul!(reshape(Y, (size(Y)..., 1)), reshape(X, (size(X)..., 1)), At, false, α, β)
    return Y
end
