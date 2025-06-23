@concrete struct ReshapeableMatrix{T} <: AbstractPNMatrix{T}
    A # not a AbstractPNMatrix
    mA
    nA
    o
end

function ReshapeableMatrix(A)
    T = eltype(A)
    o = Observable(nothing)
    if A isa AbstractPNMatrix error("ReshapeableMatrix needs a `base` Matrix") end
    mA_o, nA_o = Observable.(size(A))
    onany((args...) -> notify(o), mA_o, nA_o)
    return ReshapeableMatrix{T}(A, mA_o, nA_o, o)
end

size_string(R::ReshapeableMatrix{T}) where T = "$(size(R, 1))x$(size(R, 2)) ReshapeableMatrix{$(T)}"
content_string(R::ReshapeableMatrix) = "[$(R.A[1, 1])  $(R.A[1, 2])  ...]"

function cache_with!(ws::WorkspaceCache, cached, R::ReshapeableMatrix, α::Number, β::Number)
    R_ = get_view(R)
    cached .= α .* R_ .+ β .* cached
end

Base.size(R::ReshapeableMatrix) = (R.mA[], R.nA[])
max_size(R::ReshapeableMatrix) = size(R.A) # maybe: (length(R.A), length(R.A)) depends on how the user reshapes

required_workspace_cache(::Union{ReshapeableMatrix, Transpose{T, <:ReshapeableMatrix{T}}}) where T = WorkspaceCache(nothing, (mul_with=0, cache_with=0))
function invalidate_cache!(::ReshapeableMatrix) end

function get_view(R::ReshapeableMatrix)
    return reshape(@view(R.A[1:R.mA[]*R.nA[]]), (R.mA[], R.nA[]))
end

function reshape!(R::ReshapeableMatrix, (mA, nA))
    @assert size(R.A, 1) >= mA && size(R.A, 2) >= nA # could maybe be relaxed
    R.mA[] = mA
    R.nA[] = nA
end

function set!(R::ReshapeableMatrix, A)
    @assert size(R.A, 1) >= size(A, 1) && size(R.A, 2) >= size(A, 2)
    R.mA[] = size(A, 1)
    R.nA[] = size(A, 2)
    R_ = get_view(R)
    R_ .= A
end

function mul_with!(::WorkspaceCache, Y::AbstractVecOrMat, R::ReshapeableMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    R_ = get_view(R)
    mul!(Y, R_, X, α, β)
end

function mul_with!(::WorkspaceCache, Y::AbstractVecOrMat, X::AbstractVecOrMat, R::ReshapeableMatrix, α::Number, β::Number)
    R_ = get_view(R)
    mul!(Y, X, R_, α, β)
end

# transpose ReshapeableMatrix
size_string(Rt::Transpose{T, <:ReshapeableMatrix{T}}) where T = "$(size(Rt, 1))x$(size(Rt, 2)) transpose(::ReshapeableMatrix{$(T)})"
content_string(Rt::Transpose{T, <:ReshapeableMatrix{T}}) where T = "[$(Rt.A[1, 1])  $(Rt.A[2, 1])  ...]"

function mul_with!(::WorkspaceCache, y::AbstractVecOrMat, Rt::Transpose{T, <:ReshapeableMatrix{T}}, x::AbstractVecOrMat, α::Number, β::Number) where T
    R_ = get_view(parent(Rt))
    mul!(Y, transpose(R_), X, α, β)
end

function mul_with!(::WorkspaceCache, y::AbstractVecOrMat, x::AbstractVecOrMat, Rt::Transpose{T, <:ReshapeableMatrix{T}}, α::Number, β::Number) where T
    R_ = get_view(parent(Rt))
    mul!(Y, X, transpose(R_), α, β)
end
