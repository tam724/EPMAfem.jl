
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

Base.:*(L::AbstractLazyMatrixOrTranspose, α::Number) = lazy(*, unwrap(L), Ref(α))
Base.:*(L::AbstractLazyMatrixOrTranspose, α::Base.RefValue{<:Number}) = lazy(*, unwrap(L), α)
Base.:*(α::Number, L::AbstractLazyMatrixOrTranspose) = lazy(*, Ref(α), unwrap(L))
Base.:*(α::Base.RefValue{<:Number}, L::AbstractLazyMatrixOrTranspose) = lazy(*, α, unwrap(L))

Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, unwrap(A), unwrap(B))
Base.:*(As::Vararg{<:AbstractLazyMatrixOrTranspose}) = lazy(*, unwrap.(As)...)

Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, unwrap(L1), unwrap(L2))
Base.:+(A::AbstractMatrix, L::AbstractLazyMatrixOrTranspose) = lazy(+, A, unwrap(L))
Base.:+(L::AbstractLazyMatrixOrTranspose, A::AbstractMatrix) = lazy(+, unwrap(L), A)

Base.:-(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, unwrap(L1), lazy(*, Ref(eltype(L2)(-1)), unwrap(L2)))
Base.:-(A::AbstractMatrix, L::AbstractLazyMatrixOrTranspose) = lazy(+, A, lazy(*, Ref(eltype(L)(-1)), unwrap(L)))
Base.:-(L::AbstractLazyMatrixOrTranspose, A::AbstractMatrix) = lazy(+, unwrap(L), lazy(*, Ref(eltype(A)(-1)), A))

# damn I implemented a weird version of kron...
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))
lazy(::typeof(kron), A::AbstractMatrix, B::AbstractMatrix) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))

kron_AXB(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron_AXB, unwrap(A), unwrap(B))
materialize(A::AbstractLazyMatrixOrTranspose) = lazy(materialize, unwrap(A))
cache(A::AbstractLazyMatrixOrTranspose) = lazy(cache, unwrap(A))
LinearAlgebra.inv!(A::AbstractLazyMatrixOrTranspose) = lazy(LinearAlgebra.inv!, unwrap(A))

blockmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose, C::AbstractLazyMatrixOrTranspose, D::AbstractLazyMatrixOrTranspose) = lazy(blockmatrix, A, B, C, D)
function Base.hvcat(sizes::Tuple{<:Int64, <:Int64}, Ms::Vararg{<:AbstractLazyMatrixOrTranspose, 4})
    @assert sizes[1] == 2
    @assert sizes[2] == 2
    A, B, C, D = Ms
    return blockmatrix(A, B, C, D)
end

Krylov.minres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.minres, unwrap(A))
Krylov.gmres(A::AbstractLazyMatrixOrTranspose) = lazy(Krylov.gmres, unwrap(A))
Base.:\(A::AbstractLazyMatrixOrTranspose) = lazy(\, unwrap(A))

@concrete struct NotSoLazy{T} <: AbstractMatrix{T}
    A
    ws
end

notsolazy(A::AbstractLazyMatrix{T}, ws) where T = NotSoLazy{T}(A, ws)
notsolazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws) where T = transpose(NotSoLazy{T}(parent(At), ws))

function unlazy(A::AbstractLazyMatrix{T}, ws_alloc=zeros) where T
    ws_size = required_workspace(mul_with!, A)
    @info "allocating workspace of size $(ws_size)."
    ws = create_workspace(ws_size, ws_alloc)
    return NotSoLazy{T}(A, ws)
end

function unlazy(At::Transpose{T, <:AbstractLazyMatrix{T}}, ws_alloc=zeros) where T
    transpose(unlazy(parent(At), ws_alloc))
end

function unlazy(As::NTuple{N, AbstractLazyMatrixOrTranspose}, ws_alloc=zeros) where N
    ws_size = 0
    for A in As
        if A isa Transpose
            ws_size = max(required_workspace(mul_with!, parent(A)), ws_size)
        else
            ws_size = max(required_workspace(mul_with!, A), ws_size)
        end
    end
    ws = create_workspace(ws_size, ws_alloc)
    return notsolazy.(As, Ref(ws))
end
    

Base.getindex(A::NotSoLazy, i::Integer, j::Integer) = getindex(A.A, i, j)
Base.size(A::NotSoLazy) = size(A.A)

function LinearAlgebra.mul!(y::AbstractVector, A::NotSoLazy, x::AbstractVector, α::Number, β::Number)
    mul_with!(A.ws, y, A.A, x, α, β)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::NotSoLazy, X::AbstractMatrix, α::Number, β::Number)
    mul_with!(A.ws, Y, A.A, X, α, β)
    return Y
end

invalidate_cache!(A::NotSoLazy) = invalidate_cache!(A.ws)
