kron_AXB(A::AbstractMatrix, B::AbstractMatrix) = kron(transpose(B), A)
const KronMatrix{T} = LazyOpMatrix{T, typeof(kron_AXB), <:Tuple{AbstractMatrix{T}, AbstractMatrix{T}}}
@inline A(K::KronMatrix) = K.args[1]
@inline B(K::KronMatrix) = K.args[2]
Base.size(K::KronMatrix) = (size(A(K), 1)*size(B(K), 2), size(A(K), 2)*size(B(K), 1))
max_size(K::KronMatrix) = (max_size(A(K), 1)*max_size(B(K), 2), max_size(A(K), 2)*max_size(B(K), 1))

function lazy_getindex(K::KronMatrix, i::Int, j::Int)
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

function mul_with!(ws::Workspace, Y::AbstractMatrix, K::KronMatrix, X::AbstractMatrix, α::Number, β::Number)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    @assert size(Y, 2) == size(X, 2)
    χ = reshape(@view(X[:]), (nA, mB, size(X, 2)))
    μ = reshape(@view(Y[:]), (mA, nB, size(Y, 2)))

    strategy = mul_strategy(K)
    if strategy == :A_XB
        WS, rem = take_ws(ws, (nA, nB))
        @show size(X, 2)
        for i in 1:size(X, 2) # should be parallelized
            mul_with!(rem, WS, @view(χ[:, :, i]), B(K), true, false)
            mul_with!(rem, @view(μ[:, :, i]), A(K), WS, α, β)
        end
    else # strategy == :AX_B
        WS, rem = take_ws(ws, (mA, mB))
        @show size(X, 2)
        for i in 1:size(X, 2)
            mul_with!(rem, WS, A(K), @view(χ[:, :, i]), true, false)
            mul_with!(rem, @view(μ[:, :, i]), WS, B(K), α, β)
        end
    end
    return
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, Kt::Transpose{T, <:KronMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
    K = parent(Kt)
    mA, nA = size(A(K))
    mB, nB = size(B(K))

    @assert size(Y, 2) == size(X, 2)
    χ = reshape(@view(X[:]), (mA, nB, size(X, 2)))
    μ = reshape(@view(Y[:]), (nA, mB, size(Y, 2)))

    strategy = mul_strategy(K)
    if strategy == :A_XB # for transpose AtX_Bt
        WS, rem = take_ws(ws, (nA, nB))
        for i in 1:size(X, 2) # could be parallelized
            mul_with!(rem, WS, transpose(A(K)), @view(χ[:, :, i]), true, false)
            mul_with!(rem, @view(μ[:, :, i]),  WS, transpose(B(K)), α, β)
        end
    else # strategy == :AX_B then for transpose At_XBt
        WS, rem = take_ws(ws, (mA, mB))
        @show size(X, 2)
        for i in 1:size(X, 2)
            mul_with!(rem, WS, @view(χ[:, :, i]), transpose(B(K)), true, false)
            mul_with!(rem, @view(μ[:, :, i]), transpose(A(K)), WS, α, β)
        end
    end
    return
end

function required_workspace(::typeof(mul_with!), K::KronMatrix)
    mA, nA = max_size(A(K))
    mB, nB = max_size(B(K))

    strategy = mul_strategy(K)
    inner_ws = (strategy == :A_XB) ? nA*nB : mA*mB

    # we can do min here, because we choose the right mul_strategy
    return inner_ws + max(required_workspace(mul_with!, A(K)), required_workspace(mul_with!, B(K)))
end

function materialize_with(ws::Workspace, K::KronMatrix, ::Nothing)
    # what we do here is that we wrap both components into a lazy(materialized, ) and then materialize the full matrix
    A_ = materialize(A(K))
    B_ = materialize(B(K))

    K_mat, rem = structured_from_ws(ws, K)

    A_mat, rem_ = materialize_with(rem, A_, nothing)
    B_mat, _ = materialize_with(rem_, B_, nothing)
    
    # we implement 
    kron!(K_mat, transpose(B_mat), A_mat)
    return K_mat, rem
end

function materialize_with(ws::Workspace, K::KronMatrix, skeleton::AbstractMatrix)
    # what we do here is that we wrap both components into a lazy(materialized, ) and then materialize the full matrix
    A_ = materialize(A(K))
    B_ = materialize(B(K))

    A_mat, rem_ = materialize_with(ws, A_, nothing)
    B_mat, _ = materialize_with(rem_, B_, nothing)
    
    # we implement 
    kron!(skeleton, transpose(B_mat), A_mat)
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), K::KronMatrix)
    A_ = materialize(A(K))
    B_ = materialize(B(K))
    # the prod(size(K)) is guaranteed to be there! (by the MaterializedMatrix) we only report what we need internally
    return required_workspace(materialize_with, A_) + required_workspace(materialize_with, B_)
end
