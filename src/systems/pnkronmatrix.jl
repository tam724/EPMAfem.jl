@concrete struct KronMatrix{T} <: AbstractPNMatrix{T}
    A
    B
end

size_string(K::KronMatrix{T}) where T = "$(size(K)[1])x$(size(K)[2]) KronMatrix{$(T)}"
content_string(K::KronMatrix) = "[$(size_string(transpose(K.B))) ⊗  $(size_string(K.A))]"

function cache_with!(ws::WorkspaceCache, cached, K::KronMatrix, α::Number, β::Number)
    @assert α == true
    @assert β == false

    size_A, size_B = size(K.A), size(K.B)
    WS, rem = take_ws(ws, prod(size_A) + prod(size_B))
    cached_A = reshape(@view(WS[1:prod(size_A)]), size_A)
    cached_B = reshape(@view(WS[prod(size_A)+1:prod(size_A)+prod(size_B)]), size_B)
    cache_with!(rem[1], cached_A, K.A, true, false)
    cache_with!(rem[2], cached_B, K.B, true, false)
    kron!(cached, transpose(cached_B), cached_A)
end

function Base.size(K::KronMatrix)
    mA, nA = size(K.A)
    mB, nB = size(K.B)
    return (mA * nB, nA * mB)
end

function max_size(K::KronMatrix)
    mA, nA = max_size(K.A)
    mB, nB = max_size(K.B)
    return (mA * nB, nA * mB)
end

function required_cache_with_workspace(K::KronMatrix)
    return prod(max_size(K.A)) + prod(max_size(K.B))
end

function required_mul_with_workspace(K::KronMatrix)
    mA, nA = max_size(K.A)
    mB, nB = max_size(K.B)

    # we can do min here, because we choose the right mul_strategy
    return min(nA*nB, mA*mB)
end

function required_workspace_cache(K::KronMatrix)
    K_mul_with_ws = required_mul_with_workspace(K)
    K_cache_with_ws = required_cache_with_workspace(K)

    AB_wsch = required_workspace_cache.((K.A, K.B))

    AB_mul_with_ws = maximum(mul_with_ws, AB_wsch)
    AB_cache_with_ws = maximum(cache_with_ws, AB_wsch)
    
    return WorkspaceCache(ch.(AB_wsch), (mul_with= K_mul_with_ws + AB_mul_with_ws, cache_with=K_cache_with_ws + AB_cache_with_ws))
end

function invalidate_cache!(K::KronMatrix)
    invalidate_cache!(K.A)
    invalidate_cache!(K.B)
end

LinearAlgebra.isdiag(K::KronMatrix) = isdiag(K.A) && isdiag(K.B)
LinearAlgebra.issymmetric(K::KronMatrix) = issymmetric(K.A) && issymmetric(K.B)

function mul_strategy(K::KronMatrix)
    mA, nA = size(K.A)
    mB, nB = size(K.B)
    # A * (X * B) # cost (assuming dense n^3): nA*nB*(mB + mA)
    # (A * X) * B # cost (assuming dense n^3): mA*mB*(nA + nB)

    if (nA*nB)*(mA+mB) < (mA*mB)*(nA+nB)
        return :A_XB
    else
        return :AX_B
    end
end

function mul_with!(ws::WorkspaceCache, y::AbstractVector, K::KronMatrix, x::AbstractVector, α::Number, β::Number)
    mA, nA = size(K.A)
    mB, nB = size(K.B)

    X = reshape(@view(x[:]), (nA, mB))
    Y = reshape(@view(y[:]), (mA, nB))

    strategy = mul_strategy(K)
    if strategy == :A_XB
        WS, rem = take_ws(ws, (nA, nB))
        mul_with!(rem[2], WS, X, K.B, true, false)
        mul_with!(rem[1], Y, K.A, WS, α, β)
    else # strategy == :AX_B
        WS, rem = take_ws(ws, (mA, mB))
        mul_with!(rem[1], WS, K.A, X, true, false)
        mul_with!(rem[2], Y, WS, K.B, α, β)
    end
end

# transpose KronMatrix
size_string(Kt::Transpose{T, <:KronMatrix{T}}) where T = "$(size(Kt)[1])x$(size(Kt)[2]) transpose(::KronMatrix{$(T)})"
content_string(Kt::Transpose{T, <:KronMatrix{T}}) where T = "[$(size_string(parent(Kt).B)) ⊗  $(size_string(transpose(parent(Kt).A)))]"

function mul_strategy(Kt::Transpose{T, <:KronMatrix{T}}) where T
    mAt, nAt = size(transpose(parent(Kt).A))
    mBt, nBt = size(transpose(parent(Kt).B))
    # At * (X * Bt) # cost (assuming dense n^3): nAt*nBt*(mBt + mAt)
    # (At * X) * Bt # cost (assuming dense n^3): mAt*mBt*(nAt + nBt)

    if (nAt*nBt)*(mAt+mBt) < (mAt*mBt)*(nAt+nBt)
        return :At_XBt
    else
        return :AtX_Bt
    end
end

function mul_with!(ws::WorkspaceCache, y::AbstractVector, Kt::Transpose{T, <:KronMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    mAt, nAt = size(transpose(parent(Kt).A))
    mBt, nBt = size(transpose(parent(Kt).B))

    X = reshape(@view(x[:]), (nAt, mBt))
    Y = reshape(@view(y[:]), (mAt, nBt))

    strategy = mul_strategy(Kt)
    if strategy == :At_XBt
        WS, rem = take_ws(ws, (nAt, nBt))
        mul_with!(rem[2], WS, X, transpose(parent(Kt).B), true, false)
        mul_with!(rem[1], Y, transpose(parent(Kt).A), WS, α, β)
    else # strategy == :AX_B
        WS, rem = take_ws(ws, (mAt, mBt))
        mul_with!(rem[1], WS, transpose(parent(Kt).A), X, true, false)
        mul_with!(rem[2], Y, WS, transpose(parent(Kt).B), α, β)
    end
end
