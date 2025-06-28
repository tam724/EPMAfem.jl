# utils
# generally useful for size checking
function only_unique(iter)
    a, rest... = iter
    if !all(x -> x == a, rest)  error("Collection has multiple elements, must containt exactly 1 element") end
    return a
end

abstract type BroadcastMaterialize end
struct ShouldBroadcastMaterialize <: BroadcastMaterialize end
struct ShouldNotBroadcastMaterialize <: BroadcastMaterialize end

@concrete struct Workspace{VT<:AbstractVector}
    workspace::VT
    cache
end
@concrete struct WorkspaceSize{ST<:Integer}
    workspace::ST
    cache
end

function WorkspaceSize(ws, ch)
    return Workspace(ws, ch)
end

abstract type AbstractLazyMatrix{T} <: AbstractMatrix{T} end
const AbstractLazyMatrixOrTranspose{T} = Union{<:AbstractLazyMatrix{T}, Transpose{T, <:AbstractLazyMatrix{T}}}
Base.getindex(L::AbstractLazyMatrix{T}, I::CartesianIndex) where T = getindex(L, I.I...)
Base.getindex(L::AbstractLazyMatrix{T}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(L, i, j)
lazy_getindex(Lt::Transpose{T, <:AbstractLazyMatrix{<:T}}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(parent(Lt), j, i)
Base.getindex(::AbstractLazyMatrix{T}, args...) where T = error("should be defined")

lazy_objectid(L::AbstractLazyMatrix) = objectid(L) # we give each matrix an objectid for caching
lazy_objectid(::AbstractMatrix) = error("why cache base matrices?") 
lazy_objectid(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = lazy_objectid(parent(Lt)) # transpose gets the same as the parent

# generally opt out of broadcasting materialize..
broadcast_materialize(::AbstractLazyMatrixOrTranspose) = ShouldNotBroadcastMaterialize()

# whoa this is weird..
broadcast_materialize((first, rem)::Vararg{<:AbstractMatrix}) = combine_broadcast_materialize(broadcast_materialize(first), broadcast_materialize(rem))
combine_broadcast_materialize(::BroadcastMaterialize, ::BroadcastMaterialize) = ShouldNotBroadcastMaterialize()
combine_broadcast_materialize(::ShouldBroadcastMaterialize, ::ShouldBroadcastMaterialize) = ShouldBroadcastMaterialize() # the only case that survives..

isdiagonal(::AbstractLazyMatrix) = error("should be defined")
isdiagonal(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = isdiagonal(parent(Lt))
mul_with!(::Workspace, Y::AbstractVecOrMat, A::AbstractLazyMatrix, X::AbstractVecOrMat, ::Number, ::Number) = error("mul_with!(::Workspace, ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...) should be defined")
mul_with!(::Workspace, Y::AbstractMatrix, A::AbstractMatrix, X::AbstractLazyMatrix, ::Number, ::Number) = error("mul_with!(::Workspace, ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...) should be defined")
# this function should return the workspace size that is needed to mul_with! the AbstractLazyMatrix
required_workspace(::typeof(mul_with!), ::AbstractLazyMatrix) = error("should be defined")
required_workspace(::typeof(mul_with!), Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = required_workspace(mul_with!, parent(Lt))

materialize_with(::Workspace, ::AbstractLazyMatrix, from_cache=nothing) = error("should be defined")
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, from_cache=nothing) where T 
    L, rem = materialize_with(ws, parent(Lt), from_cache)
    return transpose(L), rem
end
# this function should return the workspace size for the matrix itself (prod(size(⋅)) AND the workspace size that is needed for the matrix to materialize
required_workspace(::typeof(materialize_with), ::AbstractLazyMatrix) = error("should be defined")
required_workspace(::typeof(materialize_with), Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = required_workspace(materialize_with, parent(Lt))

# default dispatch for AbstractMatrices, they should not be AbstractLazyMatrices though..
max_size(A::AbstractMatrix) = size(A)
max_size(A::AbstractMatrix, n::Integer) = size(A, n)

# indirect the isdiagonal
isdiagonal(A::AbstractMatrix) = false
isdiagonal(A::Diagonal) = true

function materialize(A::AbstractMatrix)
    if A isa AbstractLazyMatrixOrTranspose error("should not happen!") end
    return A
end

# for all "normal" matrices this is probably useful..
broadcast_materialize(::AbstractMatrix) = ShouldBroadcastMaterialize()

function mul_with!(::Workspace, Y::AbstractVecOrMat, A::AbstractMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    if Y isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    if A isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    if X isa AbstractLazyMatrixOrTranspose error("should not happen! mul_with!(.., ::$(typeof(Y)), ::$(typeof(A)), ::$(typeof(X)), ...)") end
    mul!(Y, A, X, α, β)
end

required_workspace(::typeof(mul_with!), A::AbstractMatrix) = 0

# materialize_with(ws::Workspace, A::AbstractMatrix, ::Nothing) = A, ws
materialize_with(ws::Workspace, A::AbstractMatrix, from_cache) = if !isnothing(from_cache) error("cannot take from cache") else return A, ws end
required_workspace(::typeof(materialize_with), A::AbstractMatrix) = 0

materialize_broadcasted(::Workspace, A::AbstractMatrix) = A

# allocate_with(::Workspace, A::AbstractMatrix) = error("should not be called!")
# required_workspace(::typeof(allocate_with), A::AbstractMatrix) = 0

# LinearAlgebra.mul!(y::AbstractVector, A::AbstractLazyMatrix{T}, x::AbstractVector) where T = mul!(y, A, x, true, false)
# LinearAlgebra.mul!(y::AbstractMatrix, A::AbstractLazyMatrix{T}, X::AbstractMatrix) where T = mul!(y, A, X, true, false)
# LinearAlgebra.mul!(y::AbstractVector, At::Transpose{T, <:AbstractLazyMatrix{T}}, x::AbstractVector) where T = mul!(y, At, x, true, false)
# LinearAlgebra.mul!(y::AbstractMatrix, At::Transpose{T, <:AbstractLazyMatrix{T}}, X::AbstractMatrix) where T = mul!(y, At, X, true, false)

function LinearAlgebra.mul!(y::AbstractVector, A::AbstractLazyMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
    ws = create_workspace(EPMAfem.mul_with!, A, zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, y, A, x, α, β)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::AbstractLazyMatrix{T}, X::AbstractMatrix, α::Number, β::Number) where T
    @warn "Not build for this, but we try anyways..."
    ws = create_workspace(EPMAfem.mul_with!, A, zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, Y, A, X, α, β)
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, A::AbstractLazyMatrix{T}, α::Number, β::Number) where T
    @warn "Not build for this, but we try anyways..."
    ws = create_workspace(EPMAfem.mul_with!, A, zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(A))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, Y, X, A, α, β)
    return Y
end

function LinearAlgebra.mul!(y::AbstractVector, At::Transpose{T, <:AbstractLazyMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    ws = create_workspace(EPMAfem.mul_with!, parent(At), zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, y, At, x, α, β)
    return y
end

function LinearAlgebra.mul!(y::AbstractMatrix, At::Transpose{T, <:AbstractLazyMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
    @warn "Not build for this, but we try anyways..."
    ws = create_workspace(EPMAfem.mul_with!, parent(At), zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, y, At, X, α, β)
    return y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, At::Transpose{T, <:AbstractLazyMatrix{T}}, α::Number, β::Number) where T
    @warn "Not build for this, but we try anyways..."
    ws = create_workspace(EPMAfem.mul_with!, parent(At), zeros)
    if length(ws.workspace) > 0 @warn("mul!(::$(typeof(At))) allocates zeros($(T), $(length(ws.workspace)))!") end
    mul_with!(ws, Y, X, At, α, β)
    return Y
end

# # a abstract type that does not implement an operation, but an flag, how we deal with this matrix (materialized, cached)
# abstract type MarkedLazyMatrix{T} <: AbstractLazyMatrix{T} end

@concrete struct LazyOpMatrix{T} <: AbstractLazyMatrix{T}
    op
    args
end

# so normal * (mul) works
function Base.similar(L::LazyOpMatrix{T}, TS::Type, (m, n)::Tuple{Int, Int}) where T
    # TODO: HACK! so scalemat works
    if first(L.args) isa T 
        return similar(L.args[2], TS, (m, n))
    else
        return similar(L.args[1], TS, (m, n))
    end
end

function lazy(func, args...)
	T = promote_type(eltype.(args)...)
	return LazyOpMatrix{T}(func, unwrap.(args))
end

# single argument
function lazy(func, arg::AbstractVector)
	T = promote_type(eltype.(arg)...)
	return LazyOpMatrix{T}(func, unwrap.(arg))
end

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



Base.:*(L::AbstractLazyMatrixOrTranspose, α::Number) = lazy(*, unwrap(L), α)
Base.:*(α::Number, L::AbstractLazyMatrixOrTranspose) = lazy(*, α, unwrap(L))

Base.:*(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = lazy(*, unwrap(A), unwrap(B))
Base.:*(As::Vararg{<:AbstractLazyMatrixOrTranspose}) = lazy(*, unwrap.(As)...)

Base.:+(L1::AbstractLazyMatrixOrTranspose, L2::AbstractLazyMatrixOrTranspose) = lazy(+, unwrap(L1), unwrap(L2))
Base.:+(A::AbstractMatrix, L::AbstractLazyMatrixOrTranspose) = lazy(+, A, unwrap(L))
Base.:+(L::AbstractLazyMatrixOrTranspose, A::AbstractMatrix) = lazy(+, unwrap(L), A)

# damn I implemented a weird version of kron...
LinearAlgebra.kron(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose) = transpose(lazy(kron, transpose(unwrap(B)), unwrap(A)))
materialize(A::AbstractLazyMatrixOrTranspose) = lazy(materialize, unwrap(A))
blockmatrix(A::AbstractLazyMatrixOrTranspose, B::AbstractLazyMatrixOrTranspose, C::AbstractLazyMatrixOrTranspose) = lazy(blockmatrix, A, B, C)
cache(A::AbstractLazyMatrixOrTranspose) = lazy(cache, unwrap(A))

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

function structured_from_ws(ws::Workspace, L::AbstractLazyMatrix, from_cache)
    if !isnothing(from_cache) && !haskey(ws.cache, from_cache) error("$(from_cache |> typeof) with $(string(from_cache, base=16)) not in cache") end
    if isdiagonal(L)
        if !isnothing(from_cache)
            _, L_cache = ws.cache[from_cache]
            structured_L = Diagonal(@view(L_cache[1:only_unique(size(L))]))
            return structured_L, ws
        else
            memory, rem = take_ws(ws, only_unique(size(L)))
            structured_L = Diagonal(memory)
            return structured_L, rem
        end
    end
    if !isnothing(from_cache)
        _, L_cache = ws.cache[from_cache]
        structured_L = reshape(@view(L_cache[1:prod(size(L))]), size(L))
        return structured_L, ws
    else
        return take_ws(ws, size(L))
    end
end

function structured_from_ws(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, from_cache) where T
    L, ws = structured_from_ws(ws, parent(Lt), from_cache)
    return transpose(L), ws
end

# function structured_from_cache(ws::Workspace, L::AbstractLazyMatrix)
#     valid, L_cache = ws.cache[lazy_objectid(L)]
#     if !valid
#         # recompute

#     end
#     if isdiagonal(L)
#         return Diagonal(@view(L_cache[1:only_unique(size(L))]))
#     else
#         return reshape(@view(L_cache[1:prod(size(L))]), size(L))
#     end
# end

# structured_from_cache(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = transpose(structured_from_cache(ws, parent(Lt)))











# @concrete struct LazyMatrix{T} <: AbstractLazyMatrix{T}
#     A
# end

# lazy(A::AbstractMatrix{T}) where T = LazyMatrix{T}(A)
# lazy(L::AbstractLazyMatrix) = L

# isdiagonal(L::LazyMatrix) = isdiagonal(L.A)

##################### OLD

# # abstract
# abstract type AbstractPNMatrix{T} <: AbstractMatrix{T} end

# Base.show(io::IO, A::Union{AbstractPNMatrix{T}, Transpose{<:T, <:AbstractPNMatrix{T}}}) where T = print(io, "$(size_string(A)): $(content_string(A))")
# Base.show(io::IO, ::MIME"text/plain", A::Union{AbstractPNMatrix, Transpose{<:T, <:AbstractPNMatrix{T}}}) where T = show(io, A)

# is_observable(_) = false
# is_observable(::AbstractPNMatrix) = true
# get_observable(A::AbstractPNMatrix) = A.o

# required_workspace_cache(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = required_workspace_cache(parent(At))
# invalidate_cache!(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = invalidate_cache!(parent(At))

# LinearAlgebra.isdiag(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = isdiag(parent(At))

# function Base.Matrix(A::AbstractPNMatrix{T}) where T
#     ws = allocate_workspace_cache(cache_with!, required_workspace_cache(A))
#     M = zeros(T, size(A)) # |> cu
#     cache_with!(ws, M, A, true, false)
#     return M
# end
# Base.Matrix(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = transpose(Matrix(parent(At)))
# Base.size(A::AbstractPNMatrix, d) = size(A)[d]

# function LinearAlgebra.mul!(y::AbstractVector, A::Union{<:AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix{T}}}, x::AbstractVector, α::Number, β::Number) where T
#     @warn("$(A) is allocating memory!")
#     ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
#     mul_with!(ws, y, A, x, α, β)
#     return y
# end

# function LinearAlgebra.mul!(y::AbstractMatrix, A::Union{<:AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix{T}}}, x::AbstractMatrix, α::Number, β::Number) where T
#     @warn("$(A) is allocating memory!")
#     ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
#     mul_with!(ws, y, A, x, α, β)
#     return y
# end

# # workspace and cache management
# function take_ws(ws::AbstractVector, n::Int)
#     return @view(ws[1:n]), @view(ws[n+1:end])
# end

# function take_ws(ws::AbstractVector, (n, m)::Tuple{<:Int, <:Int})
#     return reshape(@view(ws[1:n*m]), (n, m)), @view(ws[n*m+1:end])
# end

# @concrete struct WorkspaceCache
#     cache
#     workspace
# end

# mul_with_ws(wsch::WorkspaceCache) = wsch.workspace.mul_with
# cache_with_ws(wsch::WorkspaceCache) = wsch.workspace.cache_with
# ch(wsch::WorkspaceCache) = wsch.cache

# function take_ws(wsch::WorkspaceCache, n::Int)
#     workspace = @view(wsch.workspace[1:n])
#     remaining = WorkspaceCache(wsch.cache, @view(wsch.workspace[n+1:end]))
#     return workspace, remaining
# end

# function take_ws(wsch::WorkspaceCache, (n, m)::Tuple{<:Int, <:Int})
#     workspace = reshape(@view(wsch.workspace[1:n*m]), (n, m))
#     remaining = WorkspaceCache(wsch.cache, @view(wsch.workspace[n*m+1:end]))
#     return workspace, remaining
# end

# function take_ch(wsch::WorkspaceCache)
#     cache, rem_cache = wsch.cache
#     remaining = WorkspaceCache(rem_cache, wsch.workspace)
#     return cache, remaining
# end

# Base.getindex(wsch::WorkspaceCache{<:Union{Vector, Tuple}, <:AbstractVector{<:Number}}, i) = WorkspaceCache(ch(wsch)[i], wsch.workspace)

# const cache_id_counter = Ref(UInt64(0))
# next_cache_id() = (cache_id_counter[] += 1)

# allocate_cache(n::Int) = (zeros(n), next_cache_id()) # |> cu
# allocate_cache(::Nothing) = nothing
# allocate_cache(ch::Union{Vector, Tuple}) = allocate_cache.(ch)

# function mul_with!(::WorkspaceCache, _, ::AbstractPNMatrix, _, ::Number, ::Number) error("interface definition, should be implemented!") end
# function mul_with!(::WorkspaceCache, _, _, ::AbstractPNMatrix, ::Number, ::Number) error("interface definition, should be implemented!") end
# function cache_with!(::WorkspaceCache, _, ::AbstractPNMatrix, ::Number, ::Number) error("interface definition, should be implemented!") end

# function allocate_workspace_cache(::typeof(mul_with!), required::WorkspaceCache)
#     workspace = zeros(mul_with_ws(required))# |> cu
#     cache = allocate_cache(ch(required))
#     return WorkspaceCache(cache, workspace)
# end

# function allocate_workspace_cache(::typeof(cache_with!), required::WorkspaceCache)
#     workspace = zeros(cache_with_ws(required))# |> cu
#     cache = allocate_cache(ch(required))
#     return WorkspaceCache(cache, workspace)
# end

# # defaults for Matrix
# max_size(A::Matrix) = size(A)

# size_string(A::Matrix{T}) where T = "$(size(A, 1))x$(size(A, 2)) Matrix{$(T)}"
# size_string(A::Transpose{T, Matrix{T}}) where T = "$(size(A, 1))x$(size(A, 2)) transpose(::Matrix{$(T)})"
# size_string(A::Diagonal{T, Vector{T}}) where T = "$(size(A, 1))x$(size(A, 2)) Diagonal{$(T), Vector{$(T)}}"

# content_string(A::Matrix) = "[$(A[1, 1])  $(A[1, 2])  ...]"
# content_string(At::Transpose{T, <:Matrix{T}}) where T = "[$(At[1, 1])  $(At[1, 2])  ...]"

# function cache_with!(_, cached::AbstractMatrix, A::Matrix, α, β)
#     cached .= α .* A .+ β .* cached
# end

# function mul_with!(_, Y::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
#     if A isa (Union{AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix}} where T) error("$(typeof(A)) should be caught!") end
#     if B isa (Union{AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix}} where T) error("$(typeof(B)) should be caught!") end
#     return mul!(Y, A, B, α, β)
# end

# required_workspace_cache(::Union{Matrix, Transpose{T, <:Matrix{T}}}) where T = WorkspaceCache(nothing, (mul_with=0, cache_with=0))
# function invalidate_cache!(::Union{Matrix, Transpose{T, <:Matrix{T}}}) where T end

# # defaults for CUDA
# max_size(A::CuMatrix) = size(A)

# size_string(A::CuMatrix{T}) where T = "$(size(A, 1))x$(size(A, 2)) CuMatrix{$(T)}"
# size_string(A::Transpose{T, <:CuMatrix{T}}) where T = "$(size(A, 1))x$(size(A, 2)) transpose(::CuMatrix{$(T)})"
# size_string(A::Diagonal{T, <:CuVector{T}}) where T = "$(size(A, 1))x$(size(A, 2)) Diagonal{$(T), CuVector{$(T)}}"

# content_string(A::CuMatrix) = "[$(CUDA.@allowscalar A[1, 1])  $(CUDA.@allowscalar A[1, 2])  ...]"
# content_string(At::Transpose{T, <:CuMatrix{T}}) where T = "[$(CUDA.@allowscalar At[1, 1])  $(CUDA.@allowscalar At[1, 2])  ...]"

# function cache_with!(_, cached::CuMatrix, A::CuMatrix, α, β)
#     cached .= α .* A .+ β .* cached
# end

# required_workspace_cache(::Union{CuMatrix, Transpose{T, <:CuMatrix{T}}}) where T = WorkspaceCache(nothing, (mul_with=0, cache_with=0))
# function invalidate_cache!(::Union{CuMatrix, Transpose{T, <:CuMatrix{T}}}) where T end

