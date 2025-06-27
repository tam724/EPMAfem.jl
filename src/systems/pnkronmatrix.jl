const KronMatrix{T} = LazyOpMatrix{T, typeof(kron), <:Tuple{AbstractMatrix{T}, AbstractMatrix{T}}}
@inline A(K::KronMatrix) = K.args[1]
@inline B(K::KronMatrix) = K.args[2]
Base.size(K::KronMatrix) = (size(A(K), 1)*size(B(K), 2), size(A(K), 2)*size(B(K), 1))
max_size(K::KronMatrix) = (max_size(A(K), 1)*max_size(B(K), 2), max_size(A(K), 2)*max_size(B(K), 1))

function lazy_getindex(K::KronMatrix, I::Vararg{Int, 2})
    i, j = I
    m, n = size(A(K))
    p, q = size(B(K))
    α, a = divrem(i-1, m)
    β, b = divrem(j-1, n)
    return B(K)[β+1, α+1] * A(K)[a+1, b+1]
end
@inline isdiagonal(K::KronMatrix) = isdiagonal(A(K)) && isdiagonal(B(K))

function mul_strategy(K::KronMatrix)
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
function mul_with!(ws::Workspace, y::AbstractVector, K::KronMatrix, x::AbstractVector, α::Number, β::Number)
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
function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:KronMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
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
function required_workspace(::typeof(mul_with!), K::KronMatrix)
    mA, nA = max_size(A(K))
    mB, nB = max_size(B(K))

    # we can do min here, because we choose the right mul_strategy
    return min(nA*nB, mA*mB) + max(required_workspace(mul_with!, A(K)), required_workspace(mul_with!, B(K)))
end

function materialize_with(ws::Workspace, K::KronMatrix)
    # what we do here is that we wrap both components into a lazy(materialized, ) and then materialize the full matrix
    A_ = materialize(A(K))
    B_ = materialize(B(K))

    K_mat, rem = structured_from_ws(ws, K)

    A_mat, rem_ = materialize_with(rem, A_)
    B_mat, _ = materialize_with(rem_, B_)
    
    # we implement 
    kron!(K_mat, transpose(B_mat), A_mat)
    return K_mat, rem
end

function required_workspace(::typeof(materialize_with), K::KronMatrix)
    A_ = materialize(A(K))
    B_ = materialize(B(K))
    # the prod(size(K)) is guaranteed to be there! (by the MaterializedMatrix) we only report what we need internally
    return required_workspace(materialize_with, A_) + required_workspace(materialize_with, B_)
end
