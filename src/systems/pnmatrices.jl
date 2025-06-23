# abstract
abstract type AbstractPNMatrix{T} <: AbstractMatrix{T} end

Base.show(io::IO, A::Union{AbstractPNMatrix{T}, Transpose{<:T, <:AbstractPNMatrix{T}}}) where T = print(io, "$(size_string(A)): $(content_string(A))")
Base.show(io::IO, ::MIME"text/plain", A::Union{AbstractPNMatrix, Transpose{<:T, <:AbstractPNMatrix{T}}}) where T = show(io, A)

is_observable(_) = false
is_observable(::AbstractPNMatrix) = true
get_observable(A::AbstractPNMatrix) = A.o

required_workspace_cache(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = required_workspace_cache(parent(At))
invalidate_cache!(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = invalidate_cache!(parent(At))

LinearAlgebra.isdiag(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = isdiag(parent(At))

function Base.Matrix(A::AbstractPNMatrix{T}) where T
    ws = allocate_workspace_cache(cache_with!, required_workspace_cache(A))
    M = zeros(T, size(A)) # |> cu
    cache_with!(ws, M, A, true, false)
    return M
end
Base.Matrix(At::Transpose{T, <:AbstractPNMatrix{T}}) where T = transpose(Matrix(parent(At)))
Base.size(A::AbstractPNMatrix, d) = size(A)[d]

function LinearAlgebra.mul!(y::AbstractVector, A::Union{<:AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix{T}}}, x::AbstractVector, α::Number, β::Number) where T
    @warn("$(A) is allocating memory!")
    ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
    mul_with!(ws, y, A, x, α, β)
    return y
end

function LinearAlgebra.mul!(y::AbstractMatrix, A::Union{<:AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix{T}}}, x::AbstractMatrix, α::Number, β::Number) where T
    @warn("$(A) is allocating memory!")
    ws = allocate_workspace_cache(mul_with!, required_workspace_cache(A))
    mul_with!(ws, y, A, x, α, β)
    return y
end

# workspace and cache management
function take_ws(ws::AbstractVector, n::Int)
    return @view(ws[1:n]), @view(ws[n+1:end])
end

function take_ws(ws::AbstractVector, (n, m)::Tuple{<:Int, <:Int})
    return reshape(@view(ws[1:n*m]), (n, m)), @view(ws[n*m+1:end])
end

@concrete struct WorkspaceCache
    cache
    workspace
end

mul_with_ws(wsch::WorkspaceCache) = wsch.workspace.mul_with
cache_with_ws(wsch::WorkspaceCache) = wsch.workspace.cache_with
ch(wsch::WorkspaceCache) = wsch.cache

function take_ws(wsch::WorkspaceCache, n::Int)
    workspace = @view(wsch.workspace[1:n])
    remaining = WorkspaceCache(wsch.cache, @view(wsch.workspace[n+1:end]))
    return workspace, remaining
end

function take_ws(wsch::WorkspaceCache, (n, m)::Tuple{<:Int, <:Int})
    workspace = reshape(@view(wsch.workspace[1:n*m]), (n, m))
    remaining = WorkspaceCache(wsch.cache, @view(wsch.workspace[n*m+1:end]))
    return workspace, remaining
end

function take_ch(wsch::WorkspaceCache)
    cache, rem_cache = wsch.cache
    remaining = WorkspaceCache(rem_cache, wsch.workspace)
    return cache, remaining
end

Base.getindex(wsch::WorkspaceCache{<:Union{Vector, Tuple}, <:AbstractVector{<:Number}}, i) = WorkspaceCache(ch(wsch)[i], wsch.workspace)

const cache_id_counter = Ref(UInt64(0))
next_cache_id() = (cache_id_counter[] += 1)

allocate_cache(n::Int) = (zeros(n), next_cache_id()) # |> cu
allocate_cache(::Nothing) = nothing
allocate_cache(ch::Union{Vector, Tuple}) = allocate_cache.(ch)

function mul_with!(::WorkspaceCache, _, ::AbstractPNMatrix, _, ::Number, ::Number) error("interface definition, should be implemented!") end
function mul_with!(::WorkspaceCache, _, _, ::AbstractPNMatrix, ::Number, ::Number) error("interface definition, should be implemented!") end
function cache_with!(::WorkspaceCache, _, ::AbstractPNMatrix, ::Number, ::Number) error("interface definition, should be implemented!") end

function allocate_workspace_cache(::typeof(mul_with!), required::WorkspaceCache)
    workspace = zeros(mul_with_ws(required))# |> cu
    cache = allocate_cache(ch(required))
    return WorkspaceCache(cache, workspace)
end

function allocate_workspace_cache(::typeof(cache_with!), required::WorkspaceCache)
    workspace = zeros(cache_with_ws(required))# |> cu
    cache = allocate_cache(ch(required))
    return WorkspaceCache(cache, workspace)
end

# defaults for Matrix
max_size(A::Matrix) = size(A)

size_string(A::Matrix{T}) where T = "$(size(A, 1))x$(size(A, 2)) Matrix{$(T)}"
size_string(A::Transpose{T, Matrix{T}}) where T = "$(size(A, 1))x$(size(A, 2)) transpose(::Matrix{$(T)})"
size_string(A::Diagonal{T, Vector{T}}) where T = "$(size(A, 1))x$(size(A, 2)) Diagonal{$(T), Vector{$(T)}}"

content_string(A::Matrix) = "[$(A[1, 1])  $(A[1, 2])  ...]"
content_string(At::Transpose{T, <:Matrix{T}}) where T = "[$(At[1, 1])  $(At[1, 2])  ...]"

function cache_with!(_, cached::AbstractMatrix, A::Matrix, α, β)
    cached .= α .* A .+ β .* cached
end

function mul_with!(_, Y::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    if A isa (Union{AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix}} where T) error("$(typeof(A)) should be caught!") end
    if B isa (Union{AbstractPNMatrix, Transpose{T, <:AbstractPNMatrix}} where T) error("$(typeof(B)) should be caught!") end
    return mul!(Y, A, B, α, β)
end

required_workspace_cache(::Union{Matrix, Transpose{T, <:Matrix{T}}}) where T = WorkspaceCache(nothing, (mul_with=0, cache_with=0))
function invalidate_cache!(::Union{Matrix, Transpose{T, <:Matrix{T}}}) where T end

# defaults for CUDA
max_size(A::CuMatrix) = size(A)

size_string(A::CuMatrix{T}) where T = "$(size(A, 1))x$(size(A, 2)) CuMatrix{$(T)}"
size_string(A::Transpose{T, <:CuMatrix{T}}) where T = "$(size(A, 1))x$(size(A, 2)) transpose(::CuMatrix{$(T)})"
size_string(A::Diagonal{T, <:CuVector{T}}) where T = "$(size(A, 1))x$(size(A, 2)) Diagonal{$(T), CuVector{$(T)}}"

content_string(A::CuMatrix) = "[$(CUDA.@allowscalar A[1, 1])  $(CUDA.@allowscalar A[1, 2])  ...]"
content_string(At::Transpose{T, <:CuMatrix{T}}) where T = "[$(CUDA.@allowscalar At[1, 1])  $(CUDA.@allowscalar At[1, 2])  ...]"

function cache_with!(_, cached::CuMatrix, A::CuMatrix, α, β)
    cached .= α .* A .+ β .* cached
end

required_workspace_cache(::Union{CuMatrix, Transpose{T, <:CuMatrix{T}}}) where T = WorkspaceCache(nothing, (mul_with=0, cache_with=0))
function invalidate_cache!(::Union{CuMatrix, Transpose{T, <:CuMatrix{T}}}) where T end

