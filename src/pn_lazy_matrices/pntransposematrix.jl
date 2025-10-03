const TransposeMatrix{T} = Transpose{T, <:AbstractLazyMatrix}

# Transpose implements almost everything for us..
lazy_getindex(Lt::Transpose{T, <:AbstractLazyMatrix{<:T}}, i::Int, j::Int) where T = CUDA.@allowscalar lazy_getindex(parent(Lt), j, i)
lazy_objectid(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = lazy_objectid(parent(Lt)) # transpose gets the same as the parent
max_size(At::Transpose{T, <:AbstractLazyMatrix}) where T = reverse(max_size(parent(At)))
function max_size(At::Transpose{T, <:AbstractLazyMatrix}, n::Integer) where T
    if n == 1
        return max_size(parent(At), 2)
    elseif n == 2
        return max_size(parent(At), 1)
    else
        error("Dimension $n out of bounds")
    end
end
isdiagonal(Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T = isdiagonal(parent(Lt))
required_workspace(::typeof(mul_with!), Lt::Transpose{T, <:AbstractLazyMatrix{T}}, n, cache_notifier) where T = required_workspace(mul_with!, parent(Lt), n, cache_notifier)


function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}) where T
    L, rem = materialize_with(ws, parent(Lt))
    return transpose(L), rem
end
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, skeleton::AbstractMatrix) where T
    L, rem = materialize_with(ws, parent(Lt), transpose(skeleton))
    return transpose(L), rem
end
function materialize_with(ws::Workspace, Lt::Transpose{T, <:AbstractLazyMatrix{T}}, skeleton::AbstractMatrix, α::Number, β::Number) where T
    L, rem = materialize_with(ws, parent(Lt), transpose(skeleton), α, β)
    return transpose(L), rem
end
required_workspace(::typeof(materialize_with), Lt::Transpose{T, <:AbstractLazyMatrix{T}}, cache_notifier) where T = required_workspace(materialize_with, parent(Lt), cache_notifier)
