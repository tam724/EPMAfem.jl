# now this is special code for EPMA
function blockmatrix(A, B, C)
    return [A               B
            transpose(B)    C]
end

const BlockMatrix{T} = LazyOpMatrix{T, typeof(blockmatrix), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}}}
A(BM::BlockMatrix) = BM.args[1]
B(BM::BlockMatrix) = BM.args[2]
C(BM::BlockMatrix) = BM.args[3]

block_size(BM::BlockMatrix) = (
    only_unique((size(A(BM), 1), size(A(BM), 2), size(B(BM), 1))), 
    only_unique((size(C(BM), 1), size(C(BM), 2), size(B(BM), 2)))
)

# may be weaker
max_block_size(BM::BlockMatrix) = (
    only_unique((max_size(A(BM), 1), max_size(A(BM), 2), max_size(B(BM), 1))), 
    only_unique((max_size(C(BM), 1), max_size(C(BM), 2), max_size(B(BM), 2)))
)

duplicate(x) = (x, x)
Base.size(BM::BlockMatrix) = duplicate(sum(block_size(BM)))
max_size(BM::BlockMatrix) = duplicate(sum(max_block_size(BM)))
isdiagonal(BM::BlockMatrix) = false # until we support B = ZERO TYPE

function Base.getindex(BM::BlockMatrix, i::Int, j::Int)
    mA, nA = size(A(BM))

    if i <= mA && j <= nA
        return A(BM)[i, j]
    elseif i <= mA && j > nA
        return B(BM)[i, j - nA]
    elseif i > mA && j <= nA
        return B(BM)[j, i - mA] # transpose(B)
    else
        return C(BM)[i - mA, j - nA]
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, BM::BlockMatrix, x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(BM)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_with!(ws, y1, A(BM), x1, α, β)
    mul_with!(ws, y1, B(BM), x2, α, true)

    mul_with!(ws, y2, transpose(B(BM)), x1, α, β)
    mul_with!(ws, y2, C(BM), x2, α, true)
end

function mul_with!(ws::Workspace, y::AbstractVector, BMt::Transpose{T, <:BlockMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    n1, n2 = block_size(parent(BMt)) # is symmetric anyways...

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_with!(ws, y1, transpose(A(parent(BMt))), x1, α, β)
    mul_with!(ws, y1, B(parent(BMt)), x2, α, true)

    mul_with!(ws, y2, transpose(B(parent(BMt))), x1, α, β)
    mul_with!(ws, y2, transpose(C(parent(BMt))), x2, α, true)
end

required_workspace(::typeof(mul_with!), BM::BlockMatrix) = maximum(required_workspace(mul_with!, A_) for A_ in (A(BM), B(BM), C(BM)))

function materialize_with(ws::Workspace, BM::BlockMatrix)
    error("not implemented...")
end
required_workspace(::typeof(materialize_with), BM::BlockMatrix) = Inf



# """
# matrix of the form
#     Δ * ⌈ α  * A        β * B     ⌉
#         ⌊ δ * βt * Bt   δ * γ * C ⌋
# """
# @concrete struct BlockMatrix{T} <: AbstractPNMatrix{T}
#     A # square
#     B
#     C # square

#     α
#     β
#     βt
#     γ
#     δ # ::Union{Val{1}, Val{-1}}

#     Δ

#     o
# end

# function BlockMatrix(A, B, C, α, β, βt, γ, Δ)
#     T = promote_type(eltype.((A, B, C))...)
#     o = Observable(nothing)
#     if is_observable(A) on(_ -> notify(o), get_observable(A)) end
#     if is_observable(B) on(_ -> notify(o), get_observable(B)) end
#     if is_observable(C) on(_ -> notify(o), get_observable(C)) end
#     α_o, β_o, βt_o, γ_o, δ_o, Δ_o = Observable{T}.((α, β, βt, γ, 1, Δ))
#     onany((args...) -> notify(o), (α_o, β_o, βt_o, γ_o, δ_o, Δ_o)...)
#     return BlockMatrix{T}(A, B, C, α_o, β_o, βt_o, γ_o, δ_o, Δ_o, o)
# end


# size_string(B::BlockMatrix{T}) where T = "$(size(B)[1])x$(size(B)[2]) BlockMatrix{$(T)}"
# function content_string(B::BlockMatrix{T}) where T
#     a, b, bt, c = size_string.((B.A, B.B, transpose(B.B), B.C))
#     a = a * repeat(' ', maximum(length, (a, bt)) - length(a))
#     b = b * repeat(' ', maximum(length, (b, c)) - length(b))
#     bt = bt * repeat(' ', maximum(length, (a, bt)) - length(bt))
#     c = c * repeat(' ', maximum(length, (b, c)) - length(c))
#     return "\n Δ * ⌈     α * $(a)        β * $(b)⌉\n     ⌊δ * βt * $(bt)    δ * γ * $(c)⌋"
# end

# Base.Matrix(B::BlockMatrix) = B.Δ[] .* [
#     B.α[] .* Matrix(B.A) B.β[] .* Matrix(B.B)
#     B.δ[] .* B.βt[] .* transpose(Matrix(B.B)) B.δ[] .* B.γ[] .* Matrix(B.C)
# ]

# function cache_with!(ws::WorkspaceCache, cached, B::BlockMatrix, α::Number, β::Number)
#     error("Not Implemented Yet!")
# end

# block_size(B::BlockMatrix) = (size(B.A)[1], size(B.C)[1])
# max_block_size(B::BlockMatrix) = (max_size(B.A)[1], max_size(B.C)[1])

# duplicate(x) = (x, x)
# Base.size(B::BlockMatrix) = duplicate(sum(block_size(B)))
# max_size(B::BlockMatrix) = duplicate(sum(max_block_size(B)))

# function required_workspace_cache(B::BlockMatrix)
#     ABC_wsch = required_workspace_cache.((B.A, B.B, B.C))
#     ABC_mul_with_ws = maximum(mul_with_ws, ABC_wsch)
#     ABC_cache_with_ws = maximum(cache_with_ws, ABC_wsch)

#     return WorkspaceCache(ch.(ABC_wsch), (mul_with=ABC_mul_with_ws, cache_with=ABC_cache_with_ws))
# end

# function invalidate_cache!(B::BlockMatrix)
#     invalidate_cache!(B.A)
#     invalidate_cache!(B.B)
#     invalidate_cache!(B.C)
# end

# # multiplication routines for the individual blocks
# function ul_mul_with!(ws::WorkspaceCache, y1::AbstractVector, B::BlockMatrix, x1::AbstractVector, α::Number, β::Number)
#     mul_with!(ws, y1, B.A, x1, B.Δ[]*B.α[]*α, β)
# end

# function ur_mul_with!(ws::WorkspaceCache, y1::AbstractVector, B::BlockMatrix, x2::AbstractVector, α::Number, β::Number)
#     mul_with!(ws, y1, B.B, x2, B.Δ[]*B.β[]*α, β)
# end

# function ll_mul_with!(ws::WorkspaceCache, y2::AbstractVector, B::BlockMatrix, x1::AbstractVector, α::Number, β::Number)
#     mul_with!(ws, y2, transpose(B.B), x1, B.Δ[]*B.δ[]*B.βt[]*α, β)
# end

# function lr_mul_with!(ws::WorkspaceCache, y2::AbstractVector, B::BlockMatrix, x2::AbstractVector, α::Number, β::Number)
#     mul_with!(ws, y2, B.C, x2, B.Δ[]B.δ[]*B.γ[]*α, β)
# end

# function mul_with!(ws::WorkspaceCache, y::AbstractVector, B::Union{BlockMatrix, Transpose{T, <:BlockMatrix{T}}}, x::AbstractVector, α::Number, β::Number) where T
#     n1, n2 = block_size(B)

#     x1 = @view(x[1:n1])
#     x2 = @view(x[n1+1:n1+n2])

#     y1 = @view(y[1:n1])
#     y2 = @view(y[n1+1:n1+n2])

#     ul_mul_with!(ws, y1, B, x1, α, β)
#     ur_mul_with!(ws, y1, B, x2, α, true)

#     ll_mul_with!(ws, y2, B, x1, α, β)
#     lr_mul_with!(ws, y2, B, x2, α, true)
# end

# # transpose
# size_string(Bt::Transpose{T, <:BlockMatrix{T}}) where T = "$(size(Bt)[1])x$(size(Bt)[2]) transpose(::BlockMatrix{$(T)})"

# block_size(Bt::Transpose{T, <:BlockMatrix{T}}) where T = block_size(parent(Bt))
# max_block_size(Bt::Transpose{T, <:BlockMatrix{T}}) where T = block_size(parent(Bt))

# function content_string(Bt::Transpose{T, <:BlockMatrix{T}}) where T
#     B = parent(Bt)
#     a, b, bt, c = size_string.((transpose(B.A), B.B, transpose(B.B), transpose(B.C)))
#     a = a * repeat(' ', maximum(length, (a, bt)) - length(a))
#     b = b * repeat(' ', maximum(length, (b, c)) - length(b))
#     bt = bt * repeat(' ', maximum(length, (a, bt)) - length(bt))
#     c = c * repeat(' ', maximum(length, (b, c)) - length(c))
#     return "\n Δ * ⌈    α * $(a)       βt * $(b)⌉\n     ⌊δ * β * $(bt)    δ * γ * $(c)⌋"
# end

# function ul_mul_with!(ws::WorkspaceCache, y1::AbstractVector, Bt::Transpose{T, <:BlockMatrix{T}}, x1::AbstractVector, α::Number, β::Number) where T
#     B = parent(Bt)
#     mul_with!(ws, y1, transpose(B.A), x1, B.Δ[]*B.α[]*α, β)
# end

# function ur_mul_with!(ws::WorkspaceCache, y1::AbstractVector, Bt::Transpose{T, <:BlockMatrix{T}}, x2::AbstractVector, α::Number, β::Number) where T
#     B = parent(Bt)
#     mul_with!(ws, y1, B.B, x2, B.Δ[]*B.βt[]*α, β)
# end

# function ll_mul_with!(ws::WorkspaceCache, y2::AbstractVector, Bt::Transpose{T, <:BlockMatrix{T}}, x1::AbstractVector, α::Number, β::Number) where T
#     B = parent(Bt)
#     mul_with!(ws, y2, transpose(B.B), x1, B.Δ[]*B.δ[]*B.β[]*α, β)
# end

# function lr_mul_with!(ws::WorkspaceCache, y2::AbstractVector, Bt::Transpose{T, <:BlockMatrix{T}}, x2::AbstractVector, α::Number, β::Number) where T
#     B = parent(Bt)
#     mul_with!(ws, y2, transpose(B.C), x2, B.Δ[]B.δ[]*B.γ[]*α, β)
# end
