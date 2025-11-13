
lazy_simplify(::typeof(kron), L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(kron, L1, L2)
# (A ⊗ B) ⊗ (C ⊗ D) = (A ⊗ B ⊗ C ⊗ D)
lazy_simplify(::typeof(kron), L1::KronMatrix, L2::KronMatrix) = lazy(kron, As(L1)..., As(L2)...)

# (A ⊗ B) ⊗ C = (A ⊗ B ⊗ C)
lazy_simplify(::typeof(kron), L1::KronMatrix, L2::AbstractLazyMatrixOrTranspose) = lazy(kron, As(L1)..., L2)

# A ⊗ (B ⊗ C) = (A ⊗ B ⊗ C)
lazy_simplify(::typeof(kron), L1::AbstractLazyMatrixOrTranspose, L2::KronMatrix) = lazy(kron, L1, As(L2)...)

# ((a*A) ⊗ B) = a*(A ⊗ B)
lazy_simplify(::typeof(kron), L1::ScaleMatrix, L2::AbstractLazyMatrixOrTranspose) = lazy(*, _a(L1), lazy_simplify(kron, A(L1), L2))

# (A ⊗ (a*B)) = a*(A ⊗ B)
lazy_simplify(::typeof(kron), L1::AbstractLazyMatrixOrTranspose, L2::ScaleMatrix) = lazy(*, _a(L2), lazy_simplify(kron, L1, A(L2)))

# ((a*A) ⊗ (b*B)) = (a*b)*(A ⊗ B)
lazy_simplify(::typeof(kron), L1::ScaleMatrix, L2::ScaleMatrix) = lazy(*, _a(L1)*_a(L2), lazy_simplify(kron, A(L1), A(L2)))

# ((a*A) ⊗ (B ⊗ C)) = a*(A ⊗ B ⊗ C)
lazy_simplify(::typeof(kron), L1::ScaleMatrix, L2::KronMatrix) = lazy(*, _a(L1), kron(A(L1), As(L2)...))

# ((A ⊗ B) ⊗ (a*C)) = a*(A ⊗ B ⊗ C)
lazy_simplify(::typeof(kron), L1::KronMatrix, L2::ScaleMatrix) = lazy(*, _a(L2), kron(As(L1)..., A(L2)))


# PROD
lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::AbstractLazyMatrixOrTranspose) = lazy(*, a, L)
lazy_simplify(::typeof(*), A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, A, B)

# a*(b*A) = (a*b) * A
lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::ScaleMatrix) = lazy_simplify(*, a*_a(L), A(L))


function lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::BlockMatrix) 
    return [lazy_simplify(*, a, A(L)) lazy_simplify(*, a, B(L))
            lazy_simplify(*, a, C(L)) lazy_simplify(*, a, D(L))]
end

function lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::BlockDiagMatrix) 
    return [lazy_simplify(*, a, A(L)) nothing
            nothing                   lazy_simplify(*, a, B(L))]
end

# A*(B*C) = A*B*C
lazy_simplify(::typeof(*), A::AbstractLazyMatrixOrTranspose, P::ProdMatrix) = lazy(*, A, As(P)...)

# (A*B)*C = A*B*C
lazy_simplify(::typeof(*), P::ProdMatrix, A::AbstractLazyMatrixOrTranspose) = lazy(*, As(P)..., A)

# (A*B)*(C*D)
lazy_simplify(::typeof(*), A::ProdMatrix, B::ProdMatrix) = lazy(*, As(A)..., As(B)...)


# SUM
lazy_simplify(::typeof(+), L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, L1, L2)
function lazy_simplify(::typeof(+), L1::SumMatrix, L2::AbstractLazyMatrixOrTranspose)
    return lazy(+, As(L1)..., L2)
end

# [A B] + [C D] = [A+C B+D]
function lazy_simplify(::typeof(+), L1::BlockMatrix{T}, L2::BlockMatrix{T}) where T
    A1, B1, C1, D1 = blocks(L1)
    A2, B2, C2, D2 = blocks(L2)
    return lazy(blockmatrix, A1 + A2, B1 + B2, C1 + C2, D1 + D2)
end
function lazy_simplify(::typeof(+), L1::BlockMatrix{T}, L2::BlockDiagMatrix{T}) where T
    A1, B1, C1, D1 = blocks(L1)
    A2, B2 = blocks(L2)
    return lazy(blockmatrix, A1 + A2, B1, C1, D1 + B2)
end
function lazy_simplify(::typeof(+), L1::BlockDiagMatrix{T}, L2::BlockMatrix{T}) where T
    A1, B1 = blocks(L1)
    A2, B2, C2, D2 = blocks(L2)
    return lazy(blockmatrix, A1 + A2, B2, C2, B1 + D2)
end
function lazy_simplify(::typeof(+), L1::BlockDiagMatrix{T}, L2::BlockDiagMatrix{T}) where T
    A1, B1 = blocks(L1)
    A2, B2 = blocks(L2)
    return lazy(blockdiagmatrix, A1 + A2, B1 + B2)
end

function split_common_prefix(A1, A2)
    n = min(length(A1), length(A2))
    k = findfirst(i -> lazy_objectid(A1[i]) != lazy_objectid(A2[i]), 1:n)
    len = isnothing(k) ? n : k - 1
    return A1[1:len], A1[len+1:end], A2[len+1:end]
end

function split_common_suffix(A1, A2)
    n = min(length(A1), length(A2))
    k = findfirst(i -> lazy_objectid(A1[end-i+1]) != lazy_objectid(A2[end-i+1]), 1:n)
    len = isnothing(k) ? n : k - 1
    return A1[end-len+1:end], A1[1:end-len], A2[1:end-len]
end

function lazy_simplify(::typeof(+), L1::KronMatrix, L2::KronMatrix)
    A1, A2 = As(L1), As(L2)
    pref, r1p, r2p = split_common_prefix(A1, A2)
    suff, r1s, r2s = split_common_suffix(A1, A2)

    if length(pref) > length(suff)
        return lazy(kron, pref..., +(kron(r1p...), kron(r2p...)))
    elseif length(suff) > 0
        return lazy(kron, +(kron(r1s...), kron(r2s...)), suff...)
    else
        return lazy(+, L1, L2)
    end
end

const ScaleKronMatrix{T} = LazyOpMatrix{T, typeof(*), <:Tuple{<:AbstractLazyScalar{T}, KronMatrix{T}}}

function lazy_simplify(::typeof(+), L1::ScaleKronMatrix, L2::ScaleKronMatrix)
    A1, A2 = As(A(L1)), As(A(L2))
    a1, a2 = _a(L1), _a(L2)

    pref, r1p, r2p = split_common_prefix(A1, A2)
    suff, r1s, r2s = split_common_suffix(A1, A2)
    if length(pref) > length(suff)
        return lazy(kron, pref..., +(a1*kron(r1p...), a2*kron(r2p...)))
    elseif length(suff) > 0
        return lazy(kron, +(a1*kron(r1s...), a2*kron(r2s...)), suff...)
    else
        return lazy(+, L1, L2)
    end
end

function lazy_simplify(::typeof(+), L1::ScaleKronMatrix, L2::KronMatrix)
    A1, A2 = As(A(L1)), As(L2)
    a1 = _a(L1)

    pref, r1p, r2p = split_common_prefix(A1, A2)
    suff, r1s, r2s = split_common_suffix(A1, A2)
    if length(pref) > length(suff)
        return lazy(kron, pref..., +(a1*kron(r1p...), kron(r2p...)))
    elseif length(suff) > 0
        return lazy(kron, +(a1*kron(r1s...), kron(r2s...)), suff...)
    else
        return lazy(+, L1, L2)
    end
end

function lazy_simplify(::typeof(+), L1::KronMatrix, L2::ScaleKronMatrix)
    A1, A2 = As(L1), As(A(L2))
    a2 = _a(L2)

    pref, r1p, r2p = split_common_prefix(A1, A2)
    suff, r1s, r2s = split_common_suffix(A1, A2)
    if length(pref) > length(suff)
        return lazy(kron, pref..., +(kron(r1p...), a2*kron(r2p...)))
    elseif length(suff) > 0
        return lazy(kron, +(kron(r1s...), a2*kron(r2s...)), suff...)
    else
        return lazy(+, L1, L2)
    end
end
