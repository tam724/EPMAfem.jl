# some pruning

lazy(a::Number) = LazyScalar(a)
lazy(A::AbstractMatrix{T}) where T = LazyMatrix{T, typeof(A)}(A)
lazy(L::AbstractLazyMatrixOrTranspose) = L

# deprecate ? (better call lazy before)
Base.:*(a::Number, L::AbstractLazyMatrixOrTranspose{T}) where T = LazyScalar(T(a)) * L
Base.:*(L::AbstractLazyMatrixOrTranspose{T}, a::Number) where T = LazyScalar(T(a)) * L

const PNLAZYMATRICES_SIMPLIFY = false
if PNLAZYMATRICES_SIMPLIFY
    Base.:*(a::AbstractLazyScalar{T}, L::AbstractLazyMatrixOrTranspose{T}) where T = _lazy_simplify(*, a, L)
    Base.:*(L::AbstractLazyMatrixOrTranspose{T}, a::AbstractLazyScalar{T}) where T = _lazy_simplify(*, a, L)
    Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = _lazy_simplify(*, A, B)

    Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = _lazy_simplify(+, lazy_expand(L1), lazy_expand(L2))
    LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = _lazy_simplify(kron, A, B)
else
    Base.:*(a::AbstractLazyScalar{T}, L::AbstractLazyMatrixOrTranspose{T}) where T = lazy(*, a, L)
    Base.:*(L::AbstractLazyMatrixOrTranspose{T}, a::AbstractLazyScalar{T}) where T = lazy(*, a, L)
    Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, A, B)

    Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, L1, L2)
    LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron, A, B)
end

LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose) = A
Base.:-(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = L1 + (-L2)
Base.:-(L::AbstractLazyMatrixOrTranspose{T}) where T = -one(T)*L


# damn I implemented a weird version of kron...
kron_AXB(A::AbstractMatrix, B::AbstractMatrix) = kron(transpose(B), A)
kron_AXB(A::AbstractMatrix, B::AbstractVector) = kron(transpose(B), A)

# materialize and cache logic
broadcast_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(broadcast_materialize, A)
mat_with_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mat_with_materialize, A)
mul_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mul_materialize, A)

cache(A::AbstractLazyMatrix) = lazy(cache, materialize(A))
cache(M::MaterializedMatrix) = lazy(cache, M)

function decide_materialize_strategy(A::AbstractLazyMatrix)
    if should_broadcast_materialize(A) return :broadcast end
    return :mat_with
    # return :mat # TODO: still unsure about this one..
    # this is a crude heuristic! (if it is "cheaper" to multiply with the matrix than to materialize, then materialize by multiplication) TOOD: should be checked better
    mat = workspace_size(required_workspace(materialize_with, A, ()))
    mA, nA = max_size(A)
    mul = min(mA, nA) * workspace_size(required_workspace(mul_with!, A, ()))
    if mat < mul
        return :mat_with
    else
        return :mul
    end
end

function materialize(A::AbstractLazyMatrix; forced=false)
    strategy = decide_materialize_strategy(A)
    if strategy == :broadcast
        return lazy(broadcast_materialize, A)
    elseif strategy == :mat_with
        return lazy(mat_with_materialize, A)
    else # strategy == :mul
        return lazy(mul_materialize, A)
    end
end

# simplify materialize and cache for LazyMatrix
materialize(L::LazyMatrix) = L
materialize(Lt::Transpose{T, <:LazyMatrix{T}}) where T = Lt
cache(L::LazyMatrix) = L
cache(Lt::Transpose{T, <:LazyMatrix{T}}) where T = Lt

function materialize(A::AbstractMatrix)
    if A isa AbstractLazyMatrixOrTranspose
        @warn "should not happen: $(typeof(A))"
    end
    return A
end
function cache(A::AbstractMatrix)
    if A isa AbstractLazyMatrixOrTranspose
        @warn "should not happen: $(typeof(A))"
    end
    return A
end
materialize(M::Union{MaterializedMatrix{T}, Transpose{T, <:MaterializedMatrix{T}}}) where T = M
materialize(C::Union{CachedMatrix{T}, Transpose{T, <:CachedMatrix{T}}}) where T = C
cache(C::Union{CachedMatrix{T}, Transpose{T, <:CachedMatrix{T}}}) where T = C

LinearAlgebra.inv!(A::MaterializedMatrix) = lazy(LinearAlgebra.inv!, A)
# force the matrix to copy here
LinearAlgebra.inv!(A::AbstractLazyMatrixOrTranspose) = lazy(LinearAlgebra.inv!, materialize(A; forced=true))

blockmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose, C::AbstractLazyMatrixOrTranspose, D::AbstractLazyMatrixOrTranspose) = lazy(blockmatrix, A, B, C, D)
blockmatrix(A::AbstractLazyMatrixOrTranspose, ::Nothing, ::Nothing, B::AbstractLazyMatrixOrTranspose) = lazy(blockdiagmatrix, A, B)
function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, Ms::Vararg{AbstractLazyMatrixOrTranspose, 4})
    @assert sizes[1] == 2
    @assert sizes[2] == 2
    A, B, C, D = Ms
    return blockmatrix(A, B, C, D)
end
function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, A::AbstractLazyMatrixOrTranspose, ::Nothing, ::Nothing, B::AbstractLazyMatrixOrTranspose)
    @assert sizes[1] == 2
    @assert sizes[2] == 2
    return blockmatrix(A, nothing, nothing, B)
end

# blockdiagmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(blockdiagmatrix, A, B)
# function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, A::AbstractLazyMatrixOrTranspose, ::Nothing, ::Nothing, B::AbstractLazyMatrixOrTranspose)
#     @assert sizes[1] == 2
#     @assert sizes[2] == 2
#     return blockdiagmatrix(A, B)
# end

Krylov.minres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.minres, A)
Krylov.gmres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.gmres, A)
Base.:\(A::AbstractLazyMatrixOrTranspose) = lazy(\, materialize(A; forced=true))

function schur_complement(BM::BlockMatrix, solver, fast_solver)
    A, B, C, D = blocks(BM)
    D⁻¹ = fast_solver(D)
    inv_AmBD⁻¹C = solver(A - B * D⁻¹ * C)
    return lazy(schur_complement, inv_AmBD⁻¹C, B, C, D⁻¹)
end

@concrete struct NotSoLazy{T} <: AbstractMatrix{T}
    A
    ws
end

function unlazy(A::AbstractLazyMatrix{T}, ws_alloc=zeros; n=1) where T
    ws_size = required_workspace(mul_with!, A, n, ())
    isinteractive() && @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(A, ws)
end

function unlazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws_alloc=zeros; n=1) where T
    ws_size = required_workspace(mul_with!, parent(At), n, ())
    isinteractive() && @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(At, ws)
end

_recursive_required_workspace_mul(A::AbstractLazyMatrix) = required_workspace(mul_with!, A, ())
_recursive_required_workspace_mul(At::Transpose{T, <:AbstractLazyMatrix}) where T = required_workspace(mul_with!, parent(At), ())
_recursive_required_workspace_mul(a::AbstractLazyScalar) = 0
_recursive_required_workspace_mul(coll) = mapreduce(_recursive_required_workspace_mul, max, coll)

_recursive_notsolazy(A::AbstractLazyMatrix{T}, ws::Workspace) where T = NotSoLazy{T}(A, ws)
_recursive_notsolazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws::Workspace) where T = NotSoLazy{T}(At, ws)
_recursive_notsolazy(a::AbstractLazyScalar{T}, ws::Workspace) where T = NotSoLazyScalar{T}(a, ws)
_recursive_notsolazy(coll, ws::Workspace) = map(t -> _recursive_notsolazy(t, ws), coll)

function unlazy(coll, ws_alloc=zeros)
    ws_size = _recursive_required_workspace_mul(coll)
    ws = create_workspace(ws_size, ws_alloc)
    return _recursive_notsolazy(coll, ws)
end

Base.getindex(A::NotSoLazy, i::Integer, j::Integer) = getindex(A.A, i, j)
Base.size(A::NotSoLazy) = size(A.A)
LinearAlgebra.transpose(A::NotSoLazy{T}) where T = NotSoLazy{T}(transpose(A.A), A.ws)

function LinearAlgebra.mul!(y::AbstractVector, A::NotSoLazy, x::AbstractVector, α::Number, β::Number)
    mul_with!(A.ws, y, A.A, x, α, β)
    return y
end

function LinearAlgebra.mul!(y::AbstractMatrix, A::NotSoLazy, x::AbstractMatrix, α::Number, β::Number)
    mul_with!(A.ws, y, A.A, x, α, β)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::NotSoLazy, α::Number, β::Number)
    mul_with!(A.ws, Y, X, A.A, α, β)
    return Y
end

# interface for NotSolLazy{ResizeMatrix}
Base.copyto!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, A_::AbstractMatrix) where T = lazy_copyto!(R.ws, R.A, A_)
resize_copyto!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, A_::AbstractMatrix) where T = lazy_resize_copyto!(R.ws, R.A, A_)
Base.resize!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, new_size) where T = lazy_resize!(R.ws, R.A, new_size)
set_memory!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, v_::AbstractVector) where T = lazy_set_memory!(R.ws, R.A, v_)
set!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, v_::AbstractVector, new_size) where T = lazy_set!(R.ws, R.A, v_, new_size)

