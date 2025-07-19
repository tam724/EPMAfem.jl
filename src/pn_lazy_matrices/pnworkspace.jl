function create_workspace(::typeof(mul_with!), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(mul_with!, L, ())
    return create_workspace(ws_size, allocate)
end
function create_workspace(::typeof(materialize_with), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(materialize_with, L, ())
    return create_workspace(ws_size, allocate)
end
create_workspace(n::Integer, allocate) = create_workspace(WorkspaceSize(n, CacheStructure(nothing, nothing)), allocate)
function create_workspace(ws::WorkspaceSize, allocate::Function)
    workspace = allocate(ws.workspace)

    return PreallWorkspace(workspace, create_cache(ws.cache, allocate))
end

function create_cache(cs::CacheStructure, allocate::Function)
    cache = Dict{UInt64, Tuple{Base.RefValue{Bool}, typeof(allocate(0))}}()
    for (cache_key, (valid, cache_size)) in cs.cache
        cache[cache_key] = (valid, allocate(cache_size))
    end
    return Cache(cache, cs.cache_notifier)
end

function Base.:+(memory_add::Integer, ws::WorkspaceSize)
    return WorkspaceSize(ws.workspace + memory_add, ws.cache)
end
Base.:+(ws::WorkspaceSize, memory_add::Integer) = memory_add + ws

function merge_cachestructure(cs1::CacheStructure, cs2::CacheStructure)
    # combine the cache_notifiers
    cache_notifier = mergewith((a, b) -> (a..., b...), cs1.cache_notifier, cs2.cache_notifier)
    # search for cache cuplicates and delete the one in cs2 -> also switch out the cache_notifiers
    cache = cs1.cache
    for (key2, val2) in cs2.cache
        if haskey(cache, key2)
            @assert cache[key2][2] == val2[2]
            # ignore the cached value, but switch out the notifiers
            for (key_, valids_) in cache_notifier
                cache_notifier[key_] = replace(valids_, val2[1]=>cache[key2][1])
            end
        else
            cache[key2] = val2
        end
    end
    for (key_, valids_) in cache_notifier
        cache_notifier[key_] = tuple(unique(valids_)...)
    end
    return CacheStructure(cache, cache_notifier)

    cache = mergewith((a, b) -> (a[1] == b[1] && a[2] == b[2]) ? a : error("cache size differs"), cs1.cache, cs2.cache)
    cache_notifier = mergewith((a, b) -> (a..., b...), cs1.cache_notifier, cs2.cache_notifier)
    return CacheStructure(cache, cache_notifier)
end

function Base.:+(ws1::WorkspaceSize, ws2::WorkspaceSize)
    return WorkspaceSize(ws1.workspace + ws2.workspace, merge_cachestructure(ws1.cache, ws2.cache))
end

function Base.max(ws1::WorkspaceSize, ws2::WorkspaceSize)
    return WorkspaceSize(max(ws1.workspace, ws2.workspace), merge_cachestructure(ws1.cache, ws2.cache))
end

Base.max(ws1::WorkspaceSize, memory_add::Integer) = WorkspaceSize(max(ws1.workspace, memory_add), ws1.cache)
Base.max(memory_add::Integer, ws1::WorkspaceSize) = max(ws1, memory_add)

workspace_size(ws::WorkspaceSize) = ws.workspace
workspace_size(n::Int) = n


function take_ws(ws::PreallWorkspace, n::Integer)
    return @view(ws.workspace[1:n]), PreallWorkspace(@view(ws.workspace[n+1:end]), ws.cache)
end

function take_ws(ws::PreallWorkspace, (n, m)::Tuple{<:Integer, <:Integer})
    return reshape(@view(ws.workspace[1:n*m]), (n, m)), PreallWorkspace(@view(ws.workspace[n*m+1:end]), ws.cache)
end

function take_ws(ws::PreallWorkspace, (n, m, k)::Tuple{<:Integer, <:Integer, <:Integer})
    return reshape(@view(ws.workspace[1:n*m*k]), (n, m, k)), PreallWorkspace(@view(ws.workspace[n*m*k+1:end]), ws.cache)
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

function required_workspace(::typeof(structured_from_ws), A::AbstractMatrix)
    return prod(size(A))
end

function required_workspace(::typeof(structured_from_ws), A::Diagonal)
    return only_unique(size(A))
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

function invalidate_cache!(ws::Workspace)
    for (_, val) in ws.cache.cache
        valid, _ = val
        valid[] = false
    end
end

function notify_cache(ws::Workspace, v)
    v_id = lazy_objectid(v)
    if haskey(ws.cache.cache_notifier, v_id)
        for valid_ref in ws.cache.cache_notifier[v_id]
            valid_ref[] = false
        end
    end
end
