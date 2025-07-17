function create_workspace(::typeof(mul_with!), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(mul_with!, L)
    return create_workspace(ws_size, allocate)
end
function create_workspace(::typeof(materialize_with), L::AbstractLazyMatrixOrTranspose, allocate)
    ws_size = required_workspace(materialize_with, L)
    return create_workspace(ws_size, allocate)
end
create_workspace(n::Integer, allocate) = create_workspace(WorkspaceSize(n, []), allocate)
function create_workspace(ws::WorkspaceSize, allocate::Function)
    workspace = allocate(ws.workspace)
    cache = Dict{UInt64, Tuple{Base.RefValue{Bool}, typeof(workspace)}}()
    for (cache_key, cache_size) in ws.cache
        cache[cache_key] = (Ref(false), allocate(cache_size))
    end
    return PreallWorkspace(workspace, cache)
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
    for (_, val) in ws.cache
        valid, _ = val
        valid[] = false
    end
end
