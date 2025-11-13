const KronMatrix{T} = LazyOpMatrix{T, typeof(kron), <:Tuple{Vararg{<:AbstractMatrix{T}}}}
@inline As(K::KronMatrix) = K.args
# @inline As(Kt::Transpose{T, <:KronMatrix{T}}) where T = map(transpose, parent(Kt).args)
Base.size(K::KronMatrix) = (prod(A -> size(A, 1), As(K)), prod(A -> size(A, 2), As(K)))
max_size(K::KronMatrix) = (prod(A -> max_size(A, 1), As(K)), prod(A -> max_size(A, 2), As(K)))
function lazy_getindex(K::KronMatrix{T}, i::Integer, j::Integer) where T
    mx = map(A -> size(A, 1), As(K))
    nx = map(A -> size(A, 2), As(K))
    
    val = one(T)
    for (t, A) in enumerate(As(K))
        m_stride = prod(mx[t+1:end])
        n_stride = prod(nx[t+1:end])
        i_t = div(i - 1, m_stride) % mx[t] + 1
        j_t = div(j - 1, n_stride) % nx[t] + 1

        val *= A[i_t, j_t]
    end

    return val
end
isdiagonal(K::KronMatrix) = all(isdiagonal, As(K))
LinearAlgebra.transpose(K::KronMatrix) = lazy(kron, transpose.(As(K))...)

_r_view(A::AbstractArray, n...) = reshape(@view(A[1:prod(n)]), n...)

function mul_with!(ws::Workspace, y::AbstractMatrix, x::AbstractMatrix, K::KronMatrix, α::Number, β::Number)
    mx = map(A -> size(A, 1), As(K))
    nx = map(A -> size(A, 2), As(K))
    max_x = prod(max(m, n) for (m, n) in zip(mx, nx))

    buffer1, rem = take_ws(ws, max_x*size(x, 1))
    buffer2, rem = take_ws(rem, max_x*size(x, 1))

    xi = reshape(x, :, first(mx))
    Ai = first(As(K))
    yi = _r_view(buffer1, size(Ai, 2), size(xi, 1))
    mul_with!(rem, yi, transpose(Ai), transpose(xi), true, false)

    for i in 2:length(As(K))
        xi = reshape(yi, :, mx[i])
        Ai = As(K)[i]
        yi = _r_view(buffer2, size(Ai, 2), size(xi, 1))
        mul_with!(rem, yi, transpose(Ai), transpose(xi), true, false)

        buffer1, buffer2 = buffer2, buffer1
    end

    xi = reshape(yi, :, size(x, 1))
    transpose!(y, xi, α, β)
    return y
end

function mul_with!(ws::Workspace, y::AbstractMatrix, x::Transpose{T, <:AbstractMatrix{T}}, K::KronMatrix{T}, α::Number, β::Number) where T
    mul_with!(ws, transpose(y), transpose(K), transpose(x), α, β)
end

function mul_with!(ws::Workspace, y::AbstractVector, K::KronMatrix, x::AbstractVector, α::Number, β::Number)
    mx = map(A -> size(A, 1), As(K))
    nx = map(A -> size(A, 2), As(K))
    max_x = prod(max(m, n) for (m, n) in zip(mx, nx))

    buffer1, rem = take_ws(ws, max_x)

    xi = reshape(x, last(nx), :)
    Aiᵀ = transpose(last(As(K)))
    yi = _r_view(buffer1, size(xi, 2), size(Aiᵀ, 2))
    mul_with!(rem, yi, transpose(xi), Aiᵀ, true, false)

    if length(As(K)) > 2
        buffer2, rem = take_ws(rem, max_x)

        for i in length(As(K))-1:-1:2
            xi = reshape(yi, nx[i], :)
            Aiᵀ = transpose(As(K)[i])
            yi = _r_view(buffer2, size(xi, 2), size(Aiᵀ, 2))
            mul_with!(rem, yi, transpose(xi), Aiᵀ, true, false)

            buffer1, buffer2 = buffer2, buffer1
        end
    end

    xi = reshape(yi, nx[1], :)
    Aiᵀ = transpose(first(As(K)))
    yi = _r_view(y, size(xi, 2), size(Aiᵀ, 2))
    mul_with!(rem, yi, transpose(xi), Aiᵀ, α, β)
end

function mul_with!(ws::Workspace, y::AbstractMatrix, K::KronMatrix{T}, x::Transpose{T, <:AbstractMatrix{T}}, α::Number, β::Number) where T
    return mul_with!(ws, transpose(y), transpose(x), transpose(K), α, β)
end

function mul_with!(ws::Workspace, y::AbstractMatrix, K::KronMatrix, x::AbstractMatrix, α::Number, β::Number)
    if size(x, 2) == 1 return mul_with!(ws, vec(y), K, vec(x), α, β) end
    mx = map(A -> size(A, 1), As(K))
    nx = map(A -> size(A, 2), As(K))
    max_x = prod(max(m, n) for (m, n) in zip(mx, nx))

    buffer1, rem = take_ws(ws, max_x*size(x, 2))
    buffer2, rem = take_ws(rem, max_x*size(x, 2))

    xi = reshape(x, last(nx), :)
    Aiᵀ = transpose(last(As(K)))
    yi = _r_view(buffer1, size(xi, 2), size(Aiᵀ, 2))
    mul_with!(rem, yi, transpose(xi), Aiᵀ, true, false)

    for i in length(As(K))-1:-1:1
        xi = reshape(yi, nx[i], :)
        Aiᵀ = transpose(As(K)[i])
        yi = _r_view(buffer2, size(xi, 2), size(Aiᵀ, 2))
        mul_with!(rem, yi, transpose(xi), Aiᵀ, true, false)

        buffer1, buffer2 = buffer2, buffer1
    end

    xi = reshape(yi, size(x, 2), :)
    transpose!(y, xi, α, β)
    return y
end

function required_workspace(::typeof(mul_with!), K::KronMatrix, n::Integer, cache_notifier)
    mx = map(A -> size(A, 1), As(K))
    nx = map(A -> size(A, 2), As(K))
    max_x = prod(max(m, n) for (m, n) in zip(mx, nx))
    batch_dim = map(i -> n*max(prod(mx[k] for k in 1:length(mx) if k != i), prod(nx[k] for k in 1:length(mx) if k != i)), 1:length(mx))
    ws_size = max_x*n
    if length(As(K)) > 2 || n != 1
        return 2 * ws_size + maximum(required_workspace(mul_with!, transpose(A), batch_dim[i], cache_notifier) for (i, A) in enumerate(As(K)))
    else
        return ws_size + maximum(required_workspace(mul_with!, transpose(A), batch_dim[i], cache_notifier) for (i, A) in enumerate(As(K)))
    end
end

function materialize_with(ws::Workspace, K::KronMatrix, skeleton::AbstractMatrix)
    A, Bs... = As(K)

    A_mat, rem_ = materialize_with(ws, materialize(A))
    B_mat, _ = materialize_with(rem_, materialize(kron(Bs...)))
    
    kron!(skeleton, A_mat, B_mat)
    return skeleton, ws
end

function materialize_with(ws::Workspace, K::KronMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    A, Bs... = As(K)

    A_mat, rem_ = materialize_with(ws, materialize(A))
    B_mat, _ = materialize_with(rem_, materialize(kron(Bs...)))
    
    kron!(skeleton, A_mat, B_mat, α, β)
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), K::KronMatrix, cache_notifier)
    A, Bs... = As(K)
    # recursive materialization
    A_ = materialize(A)
    B_ = materialize(kron(Bs...))
    return required_workspace(materialize_with, A_, cache_notifier) + required_workspace(materialize_with, B_, cache_notifier)
end
