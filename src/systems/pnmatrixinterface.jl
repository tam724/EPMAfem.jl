
# some pruning

lazy(::typeof(+), A::SumMatrix, B::AbstractMatrix) = lazy(+, A.args..., B)
lazy(::typeof(+), A::AbstractMatrix, B::SumMatrix) = lazy(+, A, B.args...)
lazy(::typeof(+), A::SumMatrix, B::SumMatrix) = lazy(+, A.args..., B.args...)

lazy(::typeof(*), A::ProdMatrix, B::AbstractMatrix) = lazy(*, A.args..., B)
lazy(::typeof(*), A::AbstractMatrix, B::ProdMatrix) = lazy(*, A, B.args...)
lazy(::typeof(*), A::ProdMatrix, B::ProdMatrix) = lazy(*, A.args..., B.args...)

# a simple wrapper so we can work with the matrices easily
@concrete struct Lazy{T} <: AbstractLazyMatrix{T}
    A
end

lazy(A::AbstractMatrix{T}) where T = Lazy{T}(A)
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

# damn I implemented a weird version of kron...
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))
lazy(::typeof(kron), A::AbstractMatrix, B::AbstractMatrix) = transpose(lazy(kron_AXB, transpose(unwrap(B)), unwrap(A)))

kron_AXB(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(kron_AXB, unwrap(A), unwrap(B))
materialize(A::AbstractLazyMatrixOrTranspose) = lazy(materialize, unwrap(A))
blockmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose, C::AbstractLazyMatrixOrTranspose) = lazy(blockmatrix, A, B, C)
cache(A::AbstractLazyMatrixOrTranspose) = lazy(cache, unwrap(A))
LinearAlgebra.inv!(A::AbstractLazyMatrixOrTranspose) = lazy(LinearAlgebra.inv!, unwrap(A))

# workspace implementation
function create_workspace(::typeof(mul_with!), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(mul_with!, L)
    return create_workspace(ws_size, allocate)
end
function create_workspace(::typeof(materialize_with), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(materialize_with, L)
    return create_workspace(ws_size, allocate)
end
create_workspace(n::Integer, allocate) = create_workspace(WorkspaceSize(n, []), allocate)
function create_workspace(ws::WorkspaceSize, allocate)
    workspace = allocate(ws.workspace)
    cache = Dict{UInt64, Tuple{Base.RefValue{Bool}, typeof(workspace)}}()
    for (cache_key, cache_size) in ws.cache
        cache[cache_key] = (Ref(false), allocate(cache_size))
    end
    return Workspace(workspace, cache)
end

function Base.:+(memory_add::Integer, ws::WorkspaceSize)
    return WorkspaceSize(ws.workspace + memory_add, ws.cache)
end
Base.:+(ws::WorkspaceSize, memory_add::Integer) = memory_add + ws

function Base.:+(ws1::WorkspaceSize, ws2::WorkspaceSize)
    cache = mergewith((a, b) -> (a == b) ? a : error("cache size differs"), ws1.cache, ws2.cache)
    return WorkspaceSize(ws1.workspace + ws2.workspace, cache)
end

function Base.max(ws1::WorkspaceSize, ws2::WorkspaceSize)
    cache = mergewith((a, b) -> (a == b) ? a : error("cache size differs"), ws1.cache, ws2.cache)
    return WorkspaceSize(max(ws1.workspace, ws2.workspace), cache)
end

Base.max(ws1::WorkspaceSize, memory_add::Integer) = WorkspaceSize(max(ws1.workspace, memory_add), ws1.cache)
Base.max(memory_add::Integer, ws1::WorkspaceSize) = max(ws1, memory_add)


function take_ws(ws::Workspace, n::Integer)
    return @view(ws.workspace[1:n]), Workspace(@view(ws.workspace[n+1:end]), ws.cache)
end

function take_ws(ws::Workspace, (n, m)::Tuple{<:Integer, <:Integer})
    return reshape(@view(ws.workspace[1:n*m]), (n, m)), Workspace(@view(ws.workspace[n*m+1:end]), ws.cache)
end


function mat_view(v::AbstractVector, m::Integer, n::Integer)
    return reshape(@view(v[1:m*n]), (m, n))
end

function structured_mat_view(v::AbstractVector, M::AbstractMatrix)
    if isdiagonal(M)
        return Diagonal(@view(v[1:only_unique(size(M))]))
    else
        return reshape(@view(v[1:prod(size(M))]), size(M))
    end
end

function structured_from_ws(ws::Workspace, L::AbstractMatrix)
    if isdiagonal(L)
        memory, rem = take_ws(ws, only_unique(size(L)))
        structured_L = Diagonal(memory)
        return structured_L, rem
    else
        return take_ws(ws, size(L))
    end
end

function required_workspace(::typeof(structured_from_ws), L::AbstractLazyMatrix)
    if isdiagonal(L) #  we only track diagonal (thats the only thing we will need this for, not general though..)
        return only_unique(max_size(L))
    else
        return prod(max_size(L))
    end
end

function structured_from_ws(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T
    L, ws = structured_from_ws(ws, parent(Lt))
    return transpose(L), ws
end

required_workspace(::typeof(structured_from_ws), Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = required_workspace(structured_from_ws, parent(Lt))

