
# some pruning

lazy(::typeof(+), A::SumMatrix, B::AbstractLazyMatrixOrTranspose) = lazy(+, A.args..., B)
lazy(::typeof(+), A::AbstractLazyMatrixOrTranspose, B::SumMatrix) = lazy(+, A, B.args...)
lazy(::typeof(+), A::SumMatrix, B::SumMatrix) = lazy(+, A.args..., B.args...)

lazy(::typeof(*), A::ProdMatrix, B::AbstractLazyMatrixOrTranspose) = lazy(*, A.args..., B)
lazy(::typeof(*), A::AbstractLazyMatrixOrTranspose, B::ProdMatrix) = lazy(*, A, B.args...)
lazy(::typeof(*), A::ProdMatrix, B::ProdMatrix) = lazy(*, A.args..., B.args...)

lazy(::typeof(kron), A::KronMatrix, B::AbstractLazyMatrixOrTranspose) = lazy(kron, A.args..., B)
lazy(::typeof(kron), A::AbstractLazyMatrixOrTranspose, B::KronMatrix) = lazy(kron, A, B.args...)
lazy(::typeof(kron), A::KronMatrix, B::KronMatrix) = lazy(kron, A.args, B.args)


lazy(a::Number) = LazyScalar(a)
lazy(A::AbstractMatrix{T}) where T = LazyMatrix{T, typeof(A)}(A)
lazy(L::AbstractLazyMatrixOrTranspose) = L

Base.:*(L::AbstractLazyMatrixOrTranspose{T}, α::Number) where T = lazy(*, L, LazyScalar(T(α)))
Base.:*(α::Number, L::AbstractLazyMatrixOrTranspose{T}) where T = lazy(*, LazyScalar(T(α)), L)

Base.:*(L::AbstractLazyMatrixOrTranspose{T}, α::LazyScalar{T}) where T = lazy(*, L, α)
Base.:*(α::LazyScalar{T}, L::AbstractLazyMatrixOrTranspose{T}) where T = lazy(*, α, L)

Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, A, B)
Base.:*(As::Vararg{<:AbstractLazyMatrixOrTranspose}) = lazy(*, As...)

Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, L1, (L2))
Base.:-(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, L1, lazy(-, L2))
Base.:-(L::AbstractLazyMatrixOrTranspose) = lazy(-, L)

LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose) = A
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron, A, B)
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, Bs::Vararg{<:AbstractLazyMatrixOrTranspose}) = lazy(kron, A, Bs...)
# damn I implemented a weird version of kron...
kron_AXB(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron, transpose(B), A)

# materialize and cache logic
broadcast_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(broadcast_materialize, A)
mat_with_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mat_with_materialize, A)
mul_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mul_materialize, A)

cache(A::AbstractLazyMatrixOrTranspose) = lazy(cache, materialize(A))
cache(M::MaterializedMatrix) = lazy(cache, M)

function decide_materialize_strategy(A::AbstractLazyMatrixOrTranspose)
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

function materialize(A::AbstractLazyMatrixOrTranspose; forced=false)
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
materialize(L::Transpose{T, <:LazyMatrix{T}}) where T = L
cache(L::LazyMatrix) = L
cache(L::Transpose{T, <:LazyMatrix{T}}) where T = L

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
function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, Ms::Vararg{<:AbstractLazyMatrixOrTranspose, 4})
    @assert sizes[1] == 2
    @assert sizes[2] == 2
    A, B, C, D = Ms
    return blockmatrix(A, B, C, D)
end

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
    @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(A, ws)
end

function unlazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws_alloc=zeros; n=1) where T
    ws_size = required_workspace(mul_with!, parent(At), n, ())
    @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(At, ws)
end

_recursive_required_workspace_mul(A::AbstractLazyMatrix) = required_workspace(mul_with!, A, ())
_recursive_required_workspace_mul(At::Transpose{T, <:AbstractLazyMatrix}) where T = required_workspace(mul_with!, parent(At), ())
_recursive_required_workspace_mul(a::LazyScalar) = 0
_recursive_required_workspace_mul(coll) = mapreduce(_recursive_required_workspace_mul, max, coll)

_recursive_notsolazy(A::AbstractLazyMatrix{T}, ws::Workspace) where T = NotSoLazy{T}(A, ws)
_recursive_notsolazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws::Workspace) where T = NotSoLazy{T}(At, ws)
_recursive_notsolazy(a::LazyScalar{T}, ws::Workspace) where T = NotSoLazyScalar{T}(a, ws)
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

