kron_AXB(A::AbstractMatrix, B::AbstractMatrix) = kron(transpose(B), A)
const KronAXBMatrix{T} = LazyOpMatrix{T, typeof(kron_AXB), <:Tuple{AbstractMatrix{T}, AbstractMatrix{T}}}
@inline A(K::KronAXBMatrix) = K.args[1]
@inline B(K::KronAXBMatrix) = K.args[2]
Base.size(K::KronAXBMatrix) = (size(A(K), 1)*size(B(K), 2), size(A(K), 2)*size(B(K), 1))
max_size(K::KronAXBMatrix) = (max_size(A(K), 1)*max_size(B(K), 2), max_size(A(K), 2)*max_size(B(K), 1))

function lazy_getindex(K::KronAXBMatrix, i::Int, j::Int)
    m, n = size(A(K))
    p, q = size(B(K))
    α, a = divrem(i-1, m)
    β, b = divrem(j-1, n)
    return B(K)[β+1, α+1] * A(K)[a+1, b+1]
end
@inline isdiagonal(K::KronAXBMatrix) = isdiagonal(A(K)) && isdiagonal(B(K))

function mul_strategy(K::KronAXBMatrix) # TODO: should we use max_size here ?
    mA, nA = size(A(K))
    mB, nB = size(B(K))
    # A * (X * B) # cost (assuming dense n^3 algorithm): nA*nB*(mB + mA)
    # (A * X) * B # cost (assuming dense n^3 algorithm): mA*mB*(nA + nB)
    if (nA*nB)*(mA+mB) < (mA*mB)*(nA+nB)
        return :A_XB
    else
        return :AX_B
    end
end
function mul_with!(ws::Workspace, y::AbstractVector, K::KronAXBMatrix, x::AbstractVector, α::Number, β::Number)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    X = reshape(@view(x[:]), (nA, mB))
    Y = reshape(@view(y[:]), (mA, nB))

    strategy = mul_strategy(K)
    if strategy == :A_XB
        WS, rem = take_ws(ws, (nA, nB))
        mul_with!(rem, WS, X, B(K), true, false)
        mul_with!(rem, Y, A(K), WS, α, β)
    else # strategy == :AX_B
        WS, rem = take_ws(ws, (mA, mB))
        mul_with!(rem, WS, A(K), X, true, false)
        mul_with!(rem, Y, WS, B(K), α, β)
    end
    return
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:KronAXBMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    K = parent(Kt)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    X = reshape(@view(x[:]), (mA, nB))
    Y = reshape(@view(y[:]), (nA, mB))

    strategy = mul_strategy(K)
    if strategy == :A_XB # for transpose AtX_Bt
        WS, rem = take_ws(ws, (nA, nB))
        mul_with!(rem, WS, transpose(A(K)), X, true, false)
        mul_with!(rem, Y,  WS, transpose(B(K)), α, β)
    else # strategy == :AX_B then for transpose At_XBt
        WS, rem = take_ws(ws, (mA, mB))
        mul_with!(rem, WS, X, transpose(B(K)), true, false)
        mul_with!(rem, Y, transpose(A(K)), WS, α, β)
    end
    return
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, K::KronAXBMatrix, X::AbstractMatrix, α::Number, β::Number)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    @assert size(Y, 2) == size(X, 2)
    χ = reshape(@view(X[:]), (nA, mB, size(X, 2)))
    μ = reshape(@view(Y[:]), (mA, nB, size(Y, 2)))

    strategy = mul_strategy(K)
    if has_batched_mul!(A(K)) && has_batched_mul!(B(K))
        if strategy == :A_XB
            WS, rem = take_ws(ws, (nA, nB, size(X, 2)))
            batched_mul!(WS, χ, B(K), true, false)
            batched_mul!(μ, A(K), WS, α, β)
        else # strategy == :AX_B
            WS, rem = take_ws(ws, (mA, mB, size(X, 2)))
            batched_mul!(WS, A(K), χ, true, false)
            batched_mul!(μ, WS, B(K), α, β)
        end
    else
        if strategy == :A_XB
            WS, rem = take_ws(ws, (nA, nB))
            for i in 1:size(X, 2) # should be parallelized
                mul_with!(rem, WS, @view(χ[:, :, i]), B(K), true, false)
                mul_with!(rem, @view(μ[:, :, i]), A(K), WS, α, β)
            end
        else # strategy == :AX_B
            WS, rem = take_ws(ws, (mA, mB))
            for i in 1:size(X, 2)
                mul_with!(rem, WS, A(K), @view(χ[:, :, i]), true, false)
                mul_with!(rem, @view(μ[:, :, i]), WS, B(K), α, β)
            end
        end
    end
    return
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, Kt::Transpose{T, <:KronAXBMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
    K = parent(Kt)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    @assert size(Y, 2) == size(X, 2)
    χ = reshape(@view(X[:]), (mA, nB, size(X, 2)))
    μ = reshape(@view(Y[:]), (nA, mB, size(Y, 2)))

    strategy = mul_strategy(K)
    if has_batched_mul!(A(K)) && has_batched_mul!(B(K))
        if strategy == :A_XB # for transpose AtX_Bt
            WS, rem = take_ws(ws, (nA, nB, size(X, 2)))
            batched_mul!(WS, transpose(A(K)), χ, true, false)
            batched_mul!(μ,  WS, transpose(B(K)), α, β)
        else # strategy == :AX_B then for transpose At_XBt
            WS, rem = take_ws(ws, (mA, mB, size(X, 2)))
            batched_mul!(WS, χ, transpose(B(K)), true, false)
            batched_mul!(μ, transpose(A(K)), WS, α, β)
        end
    else
        if strategy == :A_XB # for transpose AtX_Bt
            WS, rem = take_ws(ws, (nA, nB))
            for i in 1:size(X, 2) # could be parallelized
                mul_with!(rem, WS, transpose(A(K)), @view(χ[:, :, i]), true, false)
                mul_with!(rem, @view(μ[:, :, i]),  WS, transpose(B(K)), α, β)
            end
        else # strategy == :AX_B then for transpose At_XBt
            WS, rem = take_ws(ws, (mA, mB))
            for i in 1:size(X, 2)
                mul_with!(rem, WS, @view(χ[:, :, i]), transpose(B(K)), true, false)
                mul_with!(rem, @view(μ[:, :, i]), transpose(A(K)), WS, α, β)
            end
        end
    end
    return
end

# hacky: define default
# required_workspace(::typeof(mul_with!), A::AbstractMatrix, size) = required_workspace(mul_with!, A)
has_batched_mul!(A) = false

# required matmuls
# strategy A_XB: B: (nA, mB), A: (nA, nB)
# strategy A_XB: B: (nA, nB), A: (mA, nB)

# strategy AX_B: B: (mA, mB), A: (nA, mB)
# strategy AX_B: B: (mA, nB), A: (mA, mB)

function required_workspace(::typeof(mul_with!), K::KronAXBMatrix, n, cache_notifier)
    @assert n == 1
    mA, nA = max_size(A(K))
    mB, nB = max_size(B(K))

    strategy = mul_strategy(K)
    inner_ws = (strategy == :A_XB) ? nA*nB : mA*mB

    if strategy == :A_XB
        size_A = (max(nA, mA), nB)
        size_B = (nA, max(nB, mB))
    else
        size_A = (max(nA, mA), mB)
        size_B = (mA, max(nB, mB))
    end

    return inner_ws + max(required_workspace(mul_with!, A(K), n, cache_notifier), required_workspace(mul_with!, B(K), n, cache_notifier))
end

# function required_workspace(::typeof(mul_with!), K::KronMatrix, (mx, nx))
#     # @assert mx == max_size(K, 2)
#     mA, nA = max_size(A(K))
#     mB, nB = max_size(B(K))

#     strategy = mul_strategy(K)
#     if has_batched_mul!(A(K)) && has_batched_mul!(B(K))
#         inner_ws = (strategy == :A_XB) ? nA*nB*nx : mA*mB*nx
#         return inner_ws # the batched_mul! does not need workspace right now..
#     end

#     if strategy == :A_XB
#         size_A = (max(nA, mA), nB)
#         size_B = (nA, max(nB, mB))
#     else
#         size_A = (max(nA, mA), mB)
#         size_B = (mA, max(nB, mB))
#     end


#     inner_ws = (strategy == :A_XB) ? nA*nB : mA*mB
#     return inner_ws + max(required_workspace(mul_with!, A(K), size_A), required_workspace(mul_with!, B(K), size_B))
# end

function materialize_with(ws::Workspace, K::KronAXBMatrix, skeleton::AbstractMatrix)
    # what we do here is that we wrap both components into a lazy(materialized, ) and then materialize the full matrix
    A_ = materialize(A(K))
    B_ = materialize(B(K))

    A_mat, rem_ = materialize_with(ws, A_)
    B_mat, _ = materialize_with(rem_, B_)
    
    kron!(skeleton, transpose(B_mat), A_mat)
    return skeleton, ws
end

function materialize_with(ws::Workspace, K::KronAXBMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    # what we do here is that we wrap both components into a lazy(materialized, ) and then materialize the full matrix
    A_ = materialize(A(K))
    B_ = materialize(B(K))

    A_mat, rem_ = materialize_with(ws, A_)
    B_mat, _ = materialize_with(rem_, B_)
    
    kron!(skeleton, transpose(B_mat), A_mat, α, β)
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), K::KronAXBMatrix, cache_notifier)
    A_ = materialize(A(K))
    B_ = materialize(B(K))
    # the prod(size(K)) is guaranteed to be there! (by the MaterializedMatrix) we only report what we need internally
    return required_workspace(materialize_with, A_, cache_notifier) + required_workspace(materialize_with, B_, cache_notifier)
end

## more general kron matrix
const KronMatrix{T} = LazyOpMatrix{T, typeof(kron), <:Tuple{Vararg{<:AbstractMatrix{T}}}}
@inline As(K::KronMatrix) = K.args
@inline As(Kt::Transpose{T, <:KronMatrix{T}}) where T = map(transpose, parent(Kt).args)
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

_r_view(A::AbstractArray, n...) = reshape(@view(A[1:prod(n)]), n...)

function mul_with!(ws::Workspace, y::AbstractVector, K::Union{KronMatrix, Transpose{T, <:KronMatrix{T}}}, x::AbstractVector, α::Number, β::Number) where T
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

function mul_with!(ws::Workspace, y::AbstractMatrix, K::Union{KronMatrix, <:Transpose{T, <:KronMatrix{T}}}, x::AbstractMatrix, α::Number, β::Number) where T
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
    batch_dim = map(i -> max(prod(mx[k] for k in 1:length(mx) if k != i), prod(nx[k] for k in 1:length(mx) if k != i)), 1:length(mx))
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
