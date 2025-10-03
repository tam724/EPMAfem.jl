@concrete struct AllocWorkspace{VT<:AbstractVector} <: PNLazyMatrices.Workspace{VT}
    arch
    cache
end

function PNLazyMatrices.take_ws(ws::AllocWorkspace, n::Integer)
    return allocate_vec(ws.arch, n), ws
end

function PNLazyMatrices.take_ws(ws::AllocWorkspace, (n, m)::Tuple{<:Integer, <:Integer})
    return allocate_mat(ws.arch, n, m), ws
end

function PNLazyMatrices.take_ws(ws::AllocWorkspace, (n, m, k)::Tuple{<:Integer, <:Integer, <:Integer})
    return allocate_arr(ws.arch, n, m, k), ws
end

function PNLazyMatrices.create_workspace(ws::PNLazyMatrices.WorkspaceSize, arch::PNArchitecture)
    cache = Dict{UInt64, Tuple{Base.RefValue{Bool}, vec_type(arch)}}()
    for (cache_key, cache_size) in ws.cache
        cache[cache_key] = (Ref(false), allocate_vec(arch, cache_size))
    end
    return AllocWorkspace{vec_type(arch)}(arch, cache)
end
