@concrete struct SumMatrix{T} <: AbstractPNMatrix{T}
    As
    αs
end

size_string(S::SumMatrix{T}) where T = "$(size(S)[1])x$(size(S)[2]) SumMatrix{$(T)}"
content_string(S::SumMatrix{T}) where T = "[$(size_string(S.As[1])), ...]"

function cache_with!(ws::WorkspaceCache, cached, S::SumMatrix, α::Number, β::Number)
    for i in eachindex(S.As, S.αs)
        cache_with!(ws[i], cached, S.As[i], α*S.αs[i], (i == 1) ? β : true)
    end
end

Base.size(S::SumMatrix) = size(first(S.As))
max_size(S::SumMatrix) = max_size(first(S.As))

function required_workspace_cache(S::SumMatrix)
    As_wsch = required_workspace_cache.(S.As)
    As_mul_with_ws = maximum(mul_with_ws, As_wsch)
    As_cache_with_ws = maximum(cache_with_ws, As_wsch)

    return WorkspaceCache(ch.(As_wsch), (mul_with=As_mul_with_ws, cache_with=As_cache_with_ws))
end

function invalidate_cache!(S::SumMatrix)
    for i in eachindex(S.As)
        invalidate_cache!(S.As[i])
    end
end

LinearAlgebra.isdiag(S::SumMatrix) = all(isdiag, S.As)
LinearAlgebra.issymmetric(S::SumMatrix) = all(issymmetric, S.As)

function mul_with!(ws::WorkspaceCache{<:Union{Vector, Tuple}}, y::AbstractVecOrMat, S::SumMatrix, x::AbstractVecOrMat, α::Number, β::Number)
    for i in eachindex(S.As, S.αs)
        mul_with!(ws[i], y, S.As[i], x, S.αs[i]*α, (i == 1) ? β : true)
    end
end

function mul_with!(ws::WorkspaceCache{<:Union{Vector, Tuple}}, y::AbstractVecOrMat, x::AbstractVecOrMat, S::SumMatrix, α::Number, β::Number)
    for i in eachindex(S.As, S.αs)
        mul_with!(ws[i], y, x, S.As[i], S.αs[i]*α, (i == 1) ? β : true)
    end
end

# transpose SumMatrix
size_string(St::Transpose{T, <:SumMatrix{T}}) where T = "$(size(St)[1])x$(size(St)[2]) transpose(::SumMatrix{$(T)})"
content_string(St::Transpose{T, <:SumMatrix{T}}) where T = "[$(size_string(transpose(parent(St).As[1]))), ...]"

function mul_with!(ws::WorkspaceCache{<:Union{Vector, Tuple}}, y::AbstractVecOrMat, St::Transpose{T, <:SumMatrix{T}}, x::AbstractVecOrMat, α::Number, β::Number) where T
    for i in eachindex(parent(St).As, parent(St).αs)
        mul_with!(ws[i], y, transpose(parent(St).As[i]), x, parent(St).αs[i]*α, (i == 1) ? β : true)
    end
end

function mul_with!(ws::WorkspaceCache{<:Union{Vector, Tuple}}, y::AbstractVecOrMat, x::AbstractVecOrMat, St::Transpose{T, <:SumMatrix{T}}, α::Number, β::Number) where T
    for i in eachindex(parent(St).As, parent(St).αs)
        mul_with!(ws[i], y, x, transpose(parent(St).As[i]), parent(St).αs[i]*α, (i == 1) ? β : true)
    end
end
