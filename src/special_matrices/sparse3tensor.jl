module Sparse3Tensor

using SparseArrays
using Graphs
using LinearAlgebra
using CUDA
using ConcreteStructs
using KernelAbstractions
using Adapt

include("../utils.jl")

# some overloads to enable SparseArrays and Diagonal
function SparseArrays.nonzeros(D::Diagonal)
    return D.diag
end

function nzrows(::Diagonal, j)
    return (j, )
end

function nzrows(A::SparseMatrixCSC, j)
    return (A.rowval[k] for k in nzrange(A, j))
end

struct Sparse3TensorCOO{Tv, Ti<:Integer}
    i::Vector{Ti}
    j::Vector{Ti}
    k::Vector{Ti}
    vals::Vector{Tv}
    size::Tuple{Ti, Ti, Ti}
end

Base.size(A::Sparse3TensorCOO) = A.size
Base.size(A::Sparse3TensorCOO, i) = A.size[i]
SparseArrays.nonzeros(A::Sparse3TensorCOO) = A.vals
sparsity_rate(A::Sparse3TensorCOO) = 1.0 - length(nonzeros(A))/prod(size(A))
Base.eltype(::Sparse3TensorCOO{Tv, Ti}) where {Tv, Ti} = Tv
function Base.Array(A::Sparse3TensorCOO)
    full = zeros(eltype(A), size(A))
    for (i, j, k, v) in zip(A.i, A.j, A.k, A.vals)
        full[i, j, k] = v
    end
    return full
end

function sparse3tensor(I::AbstractVector{Ti}, J::AbstractVector{Ti}, K::AbstractVector{Ti}, V::AbstractVector{Tv}, size=nothing) where {Tv, Ti<:Integer}
    @assert length(I) == length(J) == length(K) == length(V) "Lengths of I, J, K, and V must be equal. Got lengths: I=$(length(I)), J=$(length(J)), K=$(length(K)), V=$(length(V))"
    indices = 1:length(I)
    order = sortperm(indices, by=i->(K[i], J[i], I[i]))

    filtered_order = Vector{Ti}(undef, length(indices))
    vals = Vector{Tv}(undef, length(indices))
    
    last_seen = nothing
    group_accum = zero(Tv)  # Accumulator for values
    group_index = nothing
    counter = 0       # Counter for unique entries

    # loop over all the indices, filter the duplicates and accumulate the V of the duplicates in vals
    for i in order
        if last_seen != (K[i], J[i], I[i])
            if !isnothing(last_seen)
                counter += 1
                filtered_order[counter] = group_index
                vals[counter] = group_accum
            end
            last_seen = (K[i], J[i], I[i])
            group_accum = V[i]
            group_index = i
        else
            group_accum += V[i]
        end
    end

    # Add the last accumulated value
    if !isnothing(last_seen)
        counter += 1
        filtered_order[counter] = group_index
        vals[counter] = group_accum
    end

    resize!(filtered_order, counter)
    resize!(vals, counter)

    if !isnothing(size)
        @assert size[1] >= maximum(I) 
        @assert size[2] >= maximum(J) 
        @assert size[3] >= maximum(K)
    else
        # set the size to the maximum numbers (if not specified)
        size = (maximum(I, init=0), maximum(J, init=0), maximum(K, init=0))
    end

    return Sparse3TensorCOO{Tv, Ti}(I[filtered_order], J[filtered_order], K[filtered_order], vals, size)
end

function sparse3tensor(A::AbstractArray{Tv, 3}, Ti=Int64) where Tv
    n_nonzeros = 0
    for A_ in A
        if !iszero(A_)
            n_nonzeros += 1
        end
    end

    # @show "sparsity rate is $(1.0 - n_nonzeros/length(A))"
    
    I = zeros(Ti, n_nonzeros)
    J = zeros(Ti, n_nonzeros)
    K = zeros(Ti, n_nonzeros)
    V = zeros(Tv, n_nonzeros)
    counter = 0
    
    for cart_index in eachindex(IndexCartesian(), A)
        i, j, k = cart_index.I
        v = A[cart_index]
        if !iszero(v)
            counter += 1
            I[counter] = i
            J[counter] = j
            K[counter] = k
            V[counter] = v
        end
    end
    return Sparse3TensorCOO{Tv, Ti}(I, J, K, V, size(A))
end

@inline function _checksizes(A, u::AbstractVector, v::AbstractVector, w)
    @assert length(u) == size(A, 1) "Length of u must be equal to the first dimension of the tensor"
    @assert length(v) == size(A, 2) "Length of v must be equal to the second dimension of the tensor"
    @assert length(w) == size(A, 3) "Length of w must be equal to the third dimension of the tensor"
end

@inline function _checksizes(A, u::AbstractMatrix, v::AbstractMatrix, w)
    @assert size(u, 1) == size(A, 1) "Size 1 of u must be equal to the first dimension of the tensor"
    @assert size(v, 1) == size(A, 2) "Size 1 of v must be equal to the second dimension of the tensor"
    @assert size(u, 2) == size(v, 2) "Size 2 of u must be equal to size 2 of v"
    @assert length(w) == size(A, 3) "Length of w must be equal to the third dimension of the tensor"
end

function tensordot(A::Array{T, 3}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
    _checksizes(A, u, v, w)
    ret = zero(T)
    for i in 1:size(A, 1), j in 1:size(A, 2), k in 1:size(A, 3)
        ret += A[i, j, k] * u[i] * v[j] * w[k]
    end
    return ret
end

function tensordot(A::Array{T, 3}, u::Matrix{T}, v::Matrix{T}, w::Vector{T}) where T
    _checksizes(A, u, v, w)
    ret = zero(T)
    for i in 1:size(A, 1), j in 1:size(A, 2), k in 1:size(A, 3), l in 1:size(u, 2)
        ret += A[i, j, k] * u[i, l] * v[j, l] * w[k]
    end
    return ret
end

function tensordot(A::Sparse3TensorCOO{T}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
    _checksizes(A, u, v, w)
    ret = zero(T)
    for (i, j, k, val) in zip(A.i, A.j, A.k, A.vals)
        ret += val * u[i] * v[j] * w[k]
    end
    return ret
end

function tensordot(A::Sparse3TensorCOO{T}, u::Matrix{T}, v::Matrix{T}, w::Vector{T}) where T
    _checksizes(A, u, v, w)
    ret = zero(T)
    for (i, j, k, val) in zip(A.i, A.j, A.k, A.vals)
        for l in 1:size(u, 2)
            ret += val * u[i, l] * v[j, l] * w[k]
        end
    end
    return ret
end

@concrete struct Sparse3TensorSSM
	# sum of sparse matrix (tensor compression)
	skeleton
	projector
	size::NTuple{3, <:Integer}
end

Base.size(A::Sparse3TensorSSM) = A.size
Base.size(A::Sparse3TensorSSM, i) = A.size[i]
Adapt.adapt_structure(to, A::Sparse3TensorSSM) = Sparse3TensorSSM(Adapt.adapt_structure(to, A.skeleton), Adapt.adapt_structure(to, A.projector), A.size)

function find_nzval_index(::Diagonal, i, j)::Int64
    if i == j
        return i
    end
    @error "index not found"
end

function find_nzval_index(A::SparseArrays.SparseMatrixCSC, i, j)::Int64
    # idx = findfirst(v -> A.rowval[v] == i, nzrange(A, j))
    # @show idx
    idx = searchsorted(@view(A.rowval[nzrange(A, j)]), i)
    if length(idx) == 1
        return nzrange(A, j)[idx |> first]
    end
    @error "index not found"
end

function compute_coloring(I::AbstractVector{Ti}, J::AbstractVector{Ti}, n_vals) where Ti
    g_dict = Dict{Tuple{Ti, Ti}, Vector{Ti}}()
    for i in 1:n_vals
        key = (I[i], J[i])
        arr = get!(Vector{Ti}, g_dict, key)
        push!(arr, i)
    end

    # precompute the number of edges
    n_edges = 0
    for(_, ks) in g_dict
        f(n) = n*(n-1)÷2
        n_edges += f(length(ks))
    end

    edgelist = Vector{Graphs.SimpleGraphEdge{Ti}}(undef, n_edges)

    i_edge = 0
    for (_, ks) in g_dict
        for k in 1:length(ks)
            for l in k+1:length(ks)
                i_edge += 1
                edgelist[i_edge] = Graphs.SimpleGraphEdge(ks[k], ks[l])
            end
        end
    end
    g = SimpleGraph(edgelist)
    # add remaining vertices to the graph
    add_vertices!(g, n_vals - length(Graphs.vertices(g)))

    return Graphs.degree_greedy_color(g)
end

function assemble_skeleton(I::AbstractVector{Ti}, J::AbstractVector{Ti}, n_vals, S::NTuple{3, Ti}, coloring, Tv) where Ti
    Is = Ti[]
    Js = Ti[]

    for i in 1:n_vals
        if coloring.colors[i] == 1
            push!(Is, I[i])
            push!(Js, J[i])
        end
    end

    return sparse_or_diagonal(Is, Js, zeros(Tv, length(Is)), S[1], S[2])
end

function sparse_or_diagonal(Is, Js, Vs::AbstractVector{Tv}, m, n) where Tv
    if m == n && all(ij -> ij[1] == ij[2], zip(Is, Js))
        diag = zeros(Tv, m)
        for i in eachindex(Is, Vs)
            diag[Is[i]] = Vs[i]
        end
        return Diagonal(diag)
    else
        return sparse(Is, Js, Vs, m, n)
    end
end

function compute_projectors(I::AbstractVector{Ti}, J::AbstractVector{Ti}, K::AbstractVector{Ti}, V::AbstractVector{Tv}, n_vals, S::NTuple{3, Ti}, coloring, skeleton) where {Tv, Ti}
    p_Is = [Ti[] for _ in 1:coloring.num_colors]
    p_Js = [Ti[] for _ in 1:coloring.num_colors]
    p_Vs = [Tv[] for _ in 1:coloring.num_colors]

    for i in 1:n_vals
        c = coloring.colors[i]
        idx = find_nzval_index(skeleton, I[i], J[i])
        push!(p_Is[c], idx)
        push!(p_Js[c], K[i])
        push!(p_Vs[c], V[i])
    end
    return [sparse_or_diagonal(p_Is[i], p_Js[i], p_Vs[i], length(nonzeros(skeleton)), S[3]) for i in 1:coloring.num_colors]
end

function convert_to_SSM(A::Sparse3TensorCOO{Tv, Ti}, ordering=:ijk) where {Tv, Ti}
    n_vals = length(nonzeros(A))

    if ordering == :ijk
        I, J, K = A.i, A.j, A.k
        V = A.vals
        S = A.size
    elseif ordering == :kij
        I, J, K = A.k, A.i, A.j
        V = A.vals
        S = (A.size[3], A.size[1], A.size[2])
    end

    coloring = compute_coloring(I, J, n_vals)
    skeleton = assemble_skeleton(I, J, n_vals, S, coloring, Tv)
    projectors = compute_projectors(I, J, K, V, n_vals, S, coloring, skeleton)

    return Sparse3TensorSSM(skeleton, projectors, S)
end

function tensordot(A::Sparse3TensorSSM, u::AbstractVector{T}, v::AbstractVector{T}, w::AbstractVector{T}) where T
    _checksizes(A, u, v, w)
    β = false
	for pr_ in A.projector
		mul!(nonzeros(A.skeleton), pr_, w, true, β)
        β = true
	end
    return dot(u, A.skeleton, v)
end

# generalized tensordot for Matrices u and v
function tensordot(A::Sparse3TensorSSM, u::AbstractMatrix{T}, v::AbstractMatrix{T}, w::AbstractVector{T}) where T
    _checksizes(A, u, v, w)
    β = false
	for pr_ in A.projector
		mul!(nonzeros(A.skeleton), pr_, w, true, β)
        β = true
	end
    return dot(u, A.skeleton*v)
end

function _project!(B::AbstractMatrix, A::Sparse3TensorSSM, w::AbstractVector{T}) where T
    β = false
    for pr_ in A.projector
        mul!(nonzeros(B), pr_, w, true, β)
        β = true
    end
    return B
end

project!(A::Sparse3TensorSSM, w::AbstractVector{T}) where T = _project!(A.skeleton, A, w)

function contract!(Δ_w, A::Sparse3TensorSSM, u, v, α, β)
    Δ_skeleton = similar(nonzeros(A.skeleton))
    i_nz = 0
    temp = u * transpose(v)
    for j in 1:size(A.skeleton, 2)
        for i in nzrows(A.skeleton, j)
            i_nz += 1
            # Δ_skeleton[i_nz] = dot(@view(u[i, :]), @view(v[j, :]))
            Δ_skeleton[i_nz] = temp[i, j]
        end
    end
    my_rmul!(Δ_w, β)
    for pr_ in A.projector
        mul!(Δ_w, transpose(pr_), Δ_skeleton, α, true)
    end
    return Δ_w
end

function get_ijs(A::Sparse3TensorSSM)
    is = zeros(Int64, length(nonzeros(A.skeleton)))
    js = zeros(Int64, length(nonzeros(A.skeleton)))
    i_nz = 0
    for j in 1:size(A.skeleton, 2)
        for i in nzrows(A.skeleton, j)
            i_nz += 1
            is[i_nz] = i
            js[i_nz] = j
        end
    end
    return is, js
end

function special_matmul!(uv_nz, is, js, u, v, α, β)
    @kernel function special_matmul_kernel!(out, is, js, u, v, α, β)
        k = @index(Global, Linear)
        i = is[k]
        j = js[k]
        tmp = zero(Base.eltype(out))
        for l in 1:size(u, 2)
            tmp += u[i, l] * v[j, l]
        end
        out[k] = α * tmp + β * out[k]
    end
    backend = KernelAbstractions.get_backend(uv_nz)
    kernel! = special_matmul_kernel!(backend)
    kernel!(uv_nz, is, js, u, v, α, β, ndrange=length(is))
end

function contract!(Δ_w, A::Sparse3TensorSSM, uv_nz, α, β)
    my_rmul!(Δ_w, β)
    for pr_ in A.projector
        mul!(Δ_w, transpose(pr_), uv_nz, α, true)
    end
    return Δ_w
end
end
