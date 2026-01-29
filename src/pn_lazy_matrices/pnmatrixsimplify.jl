# SUM
# A + A = 2A
function lazy_simplify(::typeof(+), L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose)
    if lazy_objectid(L1) == lazy_objectid(L2)
        T1, T2 = eltype(L1), eltype(L2)
        return (lazy(one(T1)) + lazy(one(T2)))*L1
    else
        return lazy(+, L1, L2)
    end
end

# a*A + b*A = (a + b)*A
function lazy_simplify(::typeof(+), L1::ScaleMatrix, L2::ScaleMatrix)
    if lazy_objectid(A(L1)) == lazy_objectid(A(L2))
        return (_a(L1) + _a(L2))*A(L1)
    else
        return lazy(+, L1, L2)
    end
end

# a*A + A = (a + 1)*A
function lazy_simplify(::typeof(+), L1::ScaleMatrix, L2::AbstractLazyMatrixOrTranspose)
    if lazy_objectid(A(L1)) == lazy_objectid(L2)
        T2 = eltype(L2)
        return (_a(L1) + lazy(one(T2)))*A(L1)
    else
        return lazy(+, L1, L2)
    end
end

# A + a*A = (1 + a)*A
function lazy_simplify(::typeof(+), L1::AbstractLazyMatrixOrTranspose, L2::ScaleMatrix)
    if lazy_objectid(L1) == lazy_objectid(A(L2))
        T1 = eltype(L1)
        return (lazy(one(T1)) + _a(L2))*A(L2)
    else
        return lazy(+, L1, L2)
    end
end

function _lazy_simplify_sum(accum, A::AbstractLazyMatrixOrTranspose)
    for (i, T) in enumerate(accum)
        R = lazy_simplify(+, A, T)
        if !(R isa SumMatrix)
            return lazy(+, Base.setindex(accum, R, i)...)
        end
    end
    return lazy(+, accum..., A)
end
function _lazy_simplify_sum(accum, (A, As...)::Vararg{AbstractLazyMatrixOrTranspose})
    for (i, T) in enumerate(accum)
        R = lazy_simplify(+, A, T)
        if !(R isa SumMatrix)
            return _lazy_simplify_sum(Base.setindex(accum, R, i), As...)
        end
    end
    return _lazy_simplify_sum((accum..., A), As...)
end

# (A + B) + C = A + B + C
lazy_simplify(::typeof(+), A::AbstractLazyMatrixOrTranspose, S::SumMatrix) = _lazy_simplify_sum((), A, As(S)...)
lazy_simplify(::typeof(+), S::SumMatrix, A::AbstractLazyMatrixOrTranspose) = _lazy_simplify_sum((), A, As(S)...)
# fix ambiguity
lazy_simplify(::typeof(+), A::ScaleMatrix, S::SumMatrix) = _lazy_simplify_sum((), A, As(S)...)
lazy_simplify(::typeof(+), S::SumMatrix, A::ScaleMatrix) = _lazy_simplify_sum((), A, As(S)...)

# (A + B) + (C + D) = A + B + C + D
function lazy_simplify(::typeof(+), S1::SumMatrix, S2::SumMatrix)
    A, As1... = As(S1)
    return _lazy_simplify_sum((), As(S1)..., As(S2)...)
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

function split_common_prefix_suffix(A1, A2)
    n = min(length(A1), length(A2))
    k1 = findfirst(i -> lazy_objectid(A1[i]) != lazy_objectid(A2[i]), 1:n)
    isnothing(k1) && error("A1 == A2")
    len1 = k1 - 1
    k2 = findfirst(i -> lazy_objectid(A1[end-i+1]) != lazy_objectid(A2[end-i+1]), 1:n)
    isnothing(k2) && error("A1 == A2") # impossible
    len2 = k2 - 1
    return A1[1:len1], A1[end-len2+1:end], (A1[len1+1:end-len2], A2[len1+1:end-len2])
end

function lazy_simplify(::typeof(+), L1::KronMatrix, L2::KronMatrix)
    if lazy_objectid(L1) == lazy_objectid(L2)
        T1, T2 = eltype(L1), eltype(L2)
        return (lazy(one(T1)) + lazy(one(T2)))*L1
    else
        pref, suff, (r1p, r2p) = split_common_prefix_suffix(As(L1), As(L2))

        if length(pref) > 0 || length(suff) > 0
            return lazy(kron, pref..., +(kron(r1p...), kron(r2p...)), suff...)
        else
            return lazy(+, L1, L2)
        end
    end
end

const ScaleKronMatrix{T} = LazyOpMatrix{T, typeof(*), <:Tuple{<:AbstractLazyScalar{T}, KronMatrix{T}}, _NO_KWARGS}

function lazy_simplify(::typeof(+), L1::ScaleKronMatrix, L2::ScaleKronMatrix)
    a1, a2 = _a(L1), _a(L2)
    if lazy_objectid(A(L1)) == lazy_objectid(A(L2))
        return (a1 + a2)*A(L1)
    else
        pref, suff, (r1p, r2p) = split_common_prefix_suffix(As(A(L1)), As(A(L2)))

        if length(pref) > 0 || length(suff) > 0
            return lazy(kron, pref..., +(a1*kron(r1p...), a2*kron(r2p...)), suff...)
        else
            return lazy(+, L1, L2)
        end
    end
end

function lazy_simplify(::typeof(+), L1::ScaleKronMatrix, L2::KronMatrix)
    a1 = _a(L1)
    if lazy_objectid(A(L1)) == lazy_objectid(L2)
        T2 = eltype(L2)
        return (a1 + lazy(one(T2)))*A(L1)
    else
        pref, suff, (r1p, r2p) = split_common_prefix_suffix(As(A(L1)), As(L2))

        if length(pref) > 0 || length(suff) > 0
            return lazy(kron, pref..., +(a1*kron(r1p...), kron(r2p...)), suff...)
        else
            return lazy(+, L1, L2)
        end
    end
end

function lazy_simplify(::typeof(+), L1::KronMatrix, L2::ScaleKronMatrix)
    a2 = _a(L2)
    if lazy_objectid(L1) == lazy_objectid(A(L2))
        T1 = eltype(L1)
        return (lazy(one(T1)) + a2)*A(L2)
    else
        pref, suff, (r1p, r2p) = split_common_prefix_suffix(As(L1), As(A(L2)))

        if length(pref) > 0 || length(suff) > 0
            return lazy(kron, pref..., +(kron(r1p...), a2*kron(r2p...)), suff...)
        else
            return lazy(+, L1, L2)
        end
    end
end


# PROD
lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::AbstractLazyMatrixOrTranspose) = lazy(*, a, L)
lazy_simplify(::typeof(*), A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, A, B)

# a*(b*A) = (a*b) * A
lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::ScaleMatrix) = lazy_simplify(*, a*_a(L), A(L))

# a*(A + B) = a*A + a*B
lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::SumMatrix) = +((a*A for A in As(L))...)

# A*(B*C) = A*B*C
lazy_simplify(::typeof(*), A::AbstractLazyMatrixOrTranspose, P::ProdMatrix) = lazy(*, A, As(P)...)

# (A*B)*C = A*B*C
lazy_simplify(::typeof(*), P::ProdMatrix, A::AbstractLazyMatrixOrTranspose) = lazy(*, As(P)..., A)

# (A*B)*(C*D)
lazy_simplify(::typeof(*), A::ProdMatrix, B::ProdMatrix) = lazy(*, As(A)..., As(B)...)

function lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::BlockMatrix) 
    return [lazy_simplify(*, a, A(L)) lazy_simplify(*, a, B(L))
            lazy_simplify(*, a, C(L)) lazy_simplify(*, a, D(L))]
end

function lazy_simplify(::typeof(*), a::AbstractLazyScalar, L::BlockDiagMatrix) 
    return [lazy_simplify(*, a, A(L)) nothing
            nothing                   lazy_simplify(*, a, B(L))]
end


# KRON
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


## EXPAND
lazy_expand(S::AbstractLazyMatrixOrTranspose) = S

_lazy_expand_sums(A::AbstractLazyMatrixOrTranspose) = (A, )
_lazy_expand_sums(A::SumMatrix) = A.args

_lazy_expand_sum(accum, A::AbstractLazyMatrixOrTranspose) = (accum..., _lazy_expand_sums(lazy_expand(A))...)
_lazy_expand_sum(accum, (A, As...)::Vararg{AbstractLazyMatrixOrTranspose}) = _lazy_expand_sum((accum..., _lazy_expand_sums(lazy_expand(A))...), As...)
lazy_expand(S::SumMatrix) = lazy(+, _lazy_expand_sum((), S.args...)...)

# Cartesian product of two tuples (a and b)
combine(a, b) = (a, b)
combine(a::Tuple, b) = (a..., b)
combine(a, b::Tuple) = (a, b...)
combine(a::Tuple, b::Tuple) = (a..., b...)
tuple_product((a, )::Tuple{Any}, (b, )::Tuple{Any}) = (combine(a, b), )
tuple_product((a, )::Tuple{Any}, (b, bs...)::Tuple) = (combine(a, b), tuple_product((a, ), bs)...)
tuple_product((a, as...)::Tuple, (b, )::Tuple{Any}) = (combine(a, b), tuple_product(as, (b, ))...)
tuple_product((a, as...)::Tuple, (b, bs...)::Tuple) = (combine(a, b), tuple_product((a, ), bs)..., tuple_product(as, (b, ))..., tuple_product(as, bs)...)
tuple_product((a, b, c...)::Vararg{Tuple}) = (tuple_product(tuple_product(a, b), c...)) ## take that, compiler! (crazy that this runs)

function lazy_expand(K::KronMatrix)
    factors = map(A -> _lazy_expand_sums(lazy_expand(A)), As(K))
    return lazy(+, map(args -> kron(args...), tuple_product(factors...))...)
end


# TYPE STABLE SIMPLIFICATIONS
# (A + B) + C = A + B + C
lazy(::typeof(+), A::AbstractLazyMatrixOrTranspose, S::SumMatrix) = lazy(+, A, As(S)...)
lazy(::typeof(+), S::SumMatrix, A::AbstractLazyMatrixOrTranspose) = lazy(+, As(S)..., A)
lazy(::typeof(+), S1::SumMatrix, S2::SumMatrix) = lazy(+, As(S1)..., As(S2)...)

# a * b * A
lazy(::typeof(*), a::AbstractLazyScalar, S::ScaleMatrix) = lazy(*, a*_a(S), A(S))
lazy(::typeof(*), S1::ScaleMatrix, S2::ScaleMatrix) = lazy(*, _a(S1)*_a(S2), lazy(*, A(S1), A(S2)))
lazy(::typeof(*), A1::AbstractLazyMatrix, S::ScaleMatrix) = lazy(*, _a(S), lazy(*, A1, A(S)))
lazy(::typeof(*), S::ScaleMatrix, A1::AbstractLazyMatrix) = lazy(*, _a(S), lazy(*, A(S), A1))

lazy(::typeof(*), A::AbstractLazyMatrixOrTranspose, P::ProdMatrix) = lazy(*, A, As(P)...)
lazy(::typeof(*), P::ProdMatrix, A::AbstractLazyMatrixOrTranspose) = lazy(*, As(P)..., A)
lazy(::typeof(*), P1::ProdMatrix, P2::ProdMatrix) = lazy(*, As(P1)..., As(P2)...)
# fix ambiguity
lazy(::typeof(*), S::ScaleMatrix, P::ProdMatrix) = lazy(*, _a(S), lazy(*, A(S), As(P)...))
lazy(::typeof(*), P::ProdMatrix, S::ScaleMatrix) = lazy(*, _a(S), lazy(*, As(P)..., A(S)))

lazy(::typeof(kron), A::AbstractLazyMatrixOrTranspose, P::KronMatrix) = lazy(kron, A, As(P)...)
lazy(::typeof(kron), P::KronMatrix, A::AbstractLazyMatrixOrTranspose) = lazy(kron, As(P)..., A)
lazy(::typeof(kron), P1::KronMatrix, P2::KronMatrix) = lazy(kron, As(P1)..., As(P2)...)
