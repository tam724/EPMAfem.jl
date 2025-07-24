
# some pruning

lazy(::typeof(+), A::SumMatrix, B::AbstractMatrix) = lazy(+, A.args..., B)
lazy(::typeof(+), A::AbstractMatrix, B::SumMatrix) = lazy(+, A, B.args...)
lazy(::typeof(+), A::SumMatrix, B::SumMatrix) = lazy(+, A.args..., B.args...)

lazy(::typeof(*), A::ProdMatrix, B::AbstractMatrix) = lazy(*, A.args..., B)
lazy(::typeof(*), A::AbstractMatrix, B::ProdMatrix) = lazy(*, A, B.args...)
lazy(::typeof(*), A::ProdMatrix, B::ProdMatrix) = lazy(*, A.args..., B.args...)

# a simple wrapper so we can work with base matrices easily
@concrete struct Lazy{T} <: AbstractLazyMatrix{T}
    A
end

lazy(a::Number) = LazyScalar(a)
lazy(A::AbstractMatrix{T}) where T = Lazy{T}(A)
# lazy(At::Transpose{T, <:AbstractMatrix{T}}) where T = transpose(Lazy{T}(parent(At)))
lazy(L::AbstractLazyMatrixOrTranspose) = L
unwrap(A) = A
unwrap(L::Lazy) = L.A
unwrap(Lt::Transpose{T, <:Lazy{T}}) where T = transpose(unwrap(parent(Lt)))
unwrap(L::Union{<:AbstractLazyMatrix{T}, Transpose{T, <:AbstractLazyMatrix{T}}}) where T = L

Base.size(L::Lazy) = size(unwrap(L))
Base.getindex(L::Lazy, i::Int, j::Int) = CUDA.@allowscalar getindex(unwrap(L), i, j)
LinearAlgebra.transpose(L::Lazy) = lazy(transpose(L.A))
isdiagonal(L::Lazy) = isdiagonal(L.A)

Base.:*(L::AbstractLazyMatrixOrTranspose{T}, α::Number) where T = lazy(*, unwrap(L), LazyScalar(T(α)))
Base.:*(α::Number, L::AbstractLazyMatrixOrTranspose{T}) where T = lazy(*, LazyScalar(T(α)), unwrap(L))

Base.:*(L::AbstractLazyMatrixOrTranspose{T}, α::LazyScalar{T}) where T = lazy(*, unwrap(L), α)
Base.:*(α::LazyScalar{T}, L::AbstractLazyMatrixOrTranspose{T}) where T = lazy(*, α, unwrap(L))

Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, unwrap(A), unwrap(B))
Base.:*(As::Vararg{<:AbstractLazyMatrixOrTranspose}) = lazy(*, unwrap.(As)...)

Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, unwrap(L1), unwrap(L2))
Base.:+(A::AbstractMatrix, L::AbstractLazyMatrixOrTranspose) = lazy(+, A, unwrap(L))
Base.:+(L::AbstractLazyMatrixOrTranspose, A::AbstractMatrix) = lazy(+, unwrap(L), A)

Base.:-(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, unwrap(L1), lazy(-, unwrap(L2)))
Base.:-(A::AbstractMatrix, L::AbstractLazyMatrixOrTranspose) = lazy(+, A, lazy(-, unwrap(L)))
Base.:-(L::AbstractLazyMatrixOrTranspose, A::AbstractMatrix) = lazy(+, unwrap(L), lazy(-, A))
Base.:-(L::AbstractLazyMatrixOrTranspose) = lazy(-, unwrap(L))

# damn I implemented a weird version of kron...
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))
lazy(::typeof(kron), A::AbstractMatrix, B::AbstractMatrix) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))
kron_AXB(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron_AXB, unwrap(A), unwrap(B))

# materialize and cache logic
broadcast_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(broadcast_materialize, unwrap(A))
mat_with_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mat_with_materialize, unwrap(A))
mul_materialize(A::AbstractLazyMatrixOrTranspose) = lazy(mul_materialize, unwrap(A))

cache(A::AbstractLazyMatrixOrTranspose) = lazy(cache, materialize(unwrap(A)))
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

function materialize(A::AbstractLazyMatrixOrTranspose)
    strategy = decide_materialize_strategy(A)
    if strategy == :broadcast
        return lazy(broadcast_materialize, unwrap(A))
    elseif strategy == :mat_with
        return lazy(mat_with_materialize, unwrap(A))
    else # strategy == :mul
        return lazy(mul_materialize, unwrap(A))
    end
end

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


function _force_materialize(A::AbstractLazyMatrixOrTranspose)
    A_ = unwrap(A)
    if A_ isa AbstractLazyMatrixOrTranspose
        return materialize(A_)
    else
        return lazy(broadcast_materialize, A_)
    end
end


LinearAlgebra.inv!(A::MaterializedMatrix) = lazy(LinearAlgebra.inv!, unwrap(A))
# force the matrix to copy here
LinearAlgebra.inv!(A::AbstractLazyMatrixOrTranspose) = lazy(LinearAlgebra.inv!, _force_materialize(A))

blockmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose, C::AbstractLazyMatrixOrTranspose, D::AbstractLazyMatrixOrTranspose) = lazy(blockmatrix, A, B, C, D)
function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, Ms::Vararg{<:AbstractLazyMatrixOrTranspose, 4})
    @assert sizes[1] == 2
    @assert sizes[2] == 2
    A, B, C, D = Ms
    return blockmatrix(A, B, C, D)
end

Krylov.minres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.minres, unwrap(A))
Krylov.gmres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.gmres, unwrap(A))
Base.:\(A::AbstractLazyMatrixOrTranspose) = lazy(\, _force_materialize(A))

function schur_complement(BM::BlockMatrix, solver, fast_solver)
    # TODO: HACK, remove once implemented Lazy as native matrix type
    A, B, C, D = lazy.(blocks(BM))
    D⁻¹ = fast_solver(D)
    inv_AmBD⁻¹C = solver(A - B * D⁻¹ * C)
    return lazy(schur_complement, inv_AmBD⁻¹C, B, C, D⁻¹)
end

@concrete struct NotSoLazy{T} <: AbstractMatrix{T}
    A
    ws
end

# notsolazy(A::AbstractLazyMatrix{T}, ws) where T = NotSoLazy{T}(A, ws)
# notsolazy(a::LazyScalar{T}, ws) where T = NotSoLazyScalar{T}(a, ws)
# notsolazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws) where T = transpose(NotSoLazy{T}(parent(At), ws))

function unlazy(A::AbstractLazyMatrix{T}, ws_alloc=zeros) where T
    ws_size = required_workspace(mul_with!, A, ())
    @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(A, ws)
end

function unlazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws_alloc=zeros) where T
    ws_size = required_workspace(mul_with!, parent(At), ())
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

function LinearAlgebra.mul!(Y::AbstractMatrix, A::NotSoLazy, X::AbstractMatrix, α::Number, β::Number)
    mul_with!(A.ws, Y, A.A, X, α, β)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::NotSoLazy, α::Number, β::Number)
    mul_with!(A.ws, Y, X, A.A, α, β)
    return Y
end

# interface for NotSolLazy{ResizeMatrix}
Base.copyto!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, A_::AbstractMatrix) where T = copyto!(R.ws, R.A, A_)
Base.copyto!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, A_) where T = copyto!(R.ws, R.A, A_)
resize_copyto!(R::NotSoLazy{T, <:LazyResizeMatrix{T}}, A_::AbstractMatrix) where T = resize_copyto!(R.ws, R.A, A_)

