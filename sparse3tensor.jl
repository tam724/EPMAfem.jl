using SparseArrays
using Graphs
using LinearAlgebra
using CUDA

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

function sparse3tensor(I::AbstractVector{Ti}, J::AbstractVector{Ti}, K::AbstractVector{Ti}, V::AbstractVector{Tv}, size=nothing) where {Tv, Ti<:Integer}
    @assert length(I) == length(J) == length(K) == length(V) "Lengths of I, J, K, and V must be equal. Got lengths: I=$(length(I)), J=$(length(J)), K=$(length(K)), V=$(length(V))"
    indices = 1:length(I)
    order = sortperm(indices, by=i->(K[i], J[i], I[i]))

    filtered_order = Vector{Ti}(undef, length(indices))
    vals = Vector{Tv}(undef, length(indices))
    
    last_seen = nothing
    accum = zero(Tv)  # Accumulator for values
    counter = 0       # Counter for unique entries

    # loop over all the indices, filter the duplicates and accumulate the V of the duplicates in vals
    for i in order
        if last_seen != (K[i], J[i], I[i])
            if !isnothing(last_seen)
                counter += 1
                filtered_order[counter] = order[counter]
                vals[counter] = accum
            end
            last_seen = (K[i], J[i], I[i])
            accum = V[i]
        else
            accum += V[i]
        end
    end

    # Add the last accumulated value
    if !isnothing(last_seen)
        counter += 1
        filtered_order[counter] = order[counter]
        vals[counter] = accum
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

    @show "sparsity rate is $(1.0 - n_nonzeros/length(A))"
    
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

function tensordot(A::Array{T, 3}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
    @assert length(u) == size(A, 1) "Length of u must be equal to the first dimension of the tensor"
    @assert length(v) == size(A, 2) "Length of v must be equal to the second dimension of the tensor"
    @assert length(w) == size(A, 3) "Length of w must be equal to the third dimension of the tensor"
    
    ret = zero(T)
    for i in 1:size(A, 1), j in 1:size(A, 2), k in 1:size(A, 3)
        ret += A[i, j, k] * u[i] * v[j] * w[k]
    end
    return ret
end

function tensordot(A::Sparse3TensorCOO{T}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
    @assert length(u) == A.size[1] "Length of u must be equal to the first dimension of the tensor"
    @assert length(v) == A.size[2] "Length of v must be equal to the second dimension of the tensor"
    @assert length(w) == A.size[3] "Length of w must be equal to the third dimension of the tensor"
    
    ret = zero(T)
    for (i, j, k, val) in zip(A.i, A.j, A.k, A.vals)
        ret += val * u[i] * v[j] * w[k]
    end
    return ret
end

function find_nzval_index(A, i, j)::Int64
    # idx = findfirst(v -> A.rowval[v] == i, nzrange(A, j))
    # @show idx
    idx = searchsorted(@view(A.rowval[nzrange(A, j)]), i)
    if length(idx) == 1
        return nzrange(A, j)[idx |> first]
    end
    @error "index not found"
end

function convert_to_SSM(A::Sparse3TensorCOO{Tv, Ti}, ordering=:ijk) where {Tv, Ti}
    n_vals = length(nonzeros(A))
    # compute coloring

    # (val[i, j, k] * w[k])[i, j] * u[i] * v[j]
    if ordering == :ijk
        I, J, K = A.i, A.j, A.k
        S = A.size
    elseif ordering == :kij
        I, J, K = A.k, A.i, A.j
        S = (A.size[3], A.size[1], A.size[2])
    end

    g_dict = Dict{Tuple{Ti, Ti}, Vector{Ti}}()
    for i in 1:n_vals
        key = (I[i], J[i])
        if haskey(g_dict, key)
            push!(g_dict[key], i)
        else
            g_dict[key] = [i]
        end
    end

    g = SimpleGraph(n_vals)

    for ((i, j), ks) in g_dict
        for k in 1:length(ks)
            for l in k+1:length(ks)
                add_edge!(g, Edge(ks[k], ks[l]))
            end
        end
    end

    coloring = Graphs.degree_greedy_color(g)

    Is = [Ti[] for _ in 1:coloring.num_colors]
    Js = [Ti[] for _ in 1:coloring.num_colors]

    for i in 1:n_vals
        c = coloring.colors[i]
        push!(Is[c], I[i])
        push!(Js[c], J[i])
    end

    skeletons = [sparse(Is[i], Js[i], zeros(Tv, length(Is[i])), S[1], S[2]) for i in 1:coloring.num_colors]

    p_Is = [Ti[] for _ in 1:coloring.num_colors]
    p_Js = [Ti[] for _ in 1:coloring.num_colors]
    p_Vs = [Tv[] for _ in 1:coloring.num_colors]

    for i in 1:n_vals
        c = coloring.colors[i]
        idx = find_nzval_index(skeletons[c], I[i], J[i])
        push!(p_Is[c], idx)
        push!(p_Js[c], K[i])
        push!(p_Vs[c], A.vals[i])
    end

    projectors = [sparse(p_Is[i], p_Js[i], p_Vs[i], length(nonzeros(skeletons[i])), S[3]) for i in 1:coloring.num_colors]

    return Sparse3TensorSSM{Tv, Ti, SparseMatrixCSC{Tv, Ti}}(skeletons, projectors, S)
end

struct Sparse3TensorSSM{Tv, Ti<:Integer, SMT<:AbstractSparseMatrix{Tv, Ti}}
	# sum of sparse matrix (tensor compression)
	skeleton::Vector{SMT}
	projector::Vector{SMT}
	size::Tuple{Ti, Ti, Ti}
end

function CUDA.cu(A::Sparse3TensorSSM{Tv, Ti, SMT}) where {Tv, Ti, SMT}
    skeletons = [CUDA.cu(sk) for sk in A.skeleton]
    projectors = [CUDA.cu(pr) for pr in A.projector]
    @show typeof(skeletons[1])  
    return Sparse3TensorSSM{Float32, Int32, CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}}(skeletons, projectors, A.size)
end

function tensordot(A::Sparse3TensorSSM{T}, u::AbstractVector{T}, v::AbstractVector{T}, w::AbstractVector{T}) where T
	ret = zero(T)
	for (sk_, pr_) in zip(A.skeleton, A.projector)
		mul!(nonzeros(sk_), pr_, w)
		ret += dot(u, sk_, v)
	end
	return ret
end

## tests
let 
    A = rand(10, 11, 12)
    u = rand(10)
    v = rand(11)
    w = rand(12)

    A_sparse = sparse3tensor(A)
    A_SSM = convert_to_SSM(A_sparse)

    v1 = tensordot(A, u, v, w)
    v2 = tensordot(A_sparse, u, v, w)
    v3 = tensordot(A_SSM, u, v, w)

    @assert isapprox(v1, v2)
    @assert isapprox(v1, v3)
end

A = sprand(10, 10, 0.1)
A_cu = CUDA.cu(A)

A_sparse = rand(10, 11, 12) |> sparse3tensor

Asize = (125000, 125000, 125000)
n_vals = 125000*10*3
I = rand(1:Asize[1], n_vals)
J = rand(1:Asize[2], n_vals)
K = rand(1:Asize[3], n_vals)
vals = rand(n_vals)
@profview sparse3tensor(I, J, K, vals, Asize)
sparsity_rate(A_sparse)

A_SSM_uvw = convert_to_SSM2(A_sparse, :ijk)
A_SSM_uvw_cu = CUDA.cu(A_SSM_uvw)

A_SSM_wuv = convert_to_SSM2(A_sparse, :kij)
@profview convert_to_SSM2(A_sparse)

u = rand(size(A_sparse, 1))
v = rand(size(A_sparse, 2))
w = rand(size(A_sparse, 3))
u_cu, v_cu, w_cu = CUDA.cu(u), CUDA.cu(v), CUDA.cu(w)

tensordot(A_sparse, u, v, w)

@time tensordot(A_SSM_uvw, u, v, w)
@time tensordot(A_SSM_uvw_cu, u_cu, v_cu, w_cu)

tensordot(A_SSM_wuv, w, u, v)

using BenchmarkTools
@benchmark tensordot(A_sparse, u, v, w) 
@benchmark tensordot(A_SSM_uvw, u, v, w)
@benchmark tensordot(A_SSM_wuv, w, u, v)

@profview tensordot(A_sparse, u, v, w)
@profview tensordot(A_SSM, u, v, w)

begin
	i = [1, 1, 1, 1, 2, 2, 2, 2]
	j = [1, 1, 2, 2, 1, 1, 2, 2]
	k = [1, 2, 1, 2, 1, 2, 1, 2]
	vals2 = rand(8)
	
	sparse3tensor(i, j, k, vals2, nothing)
end

begin
	i = Int64[]
	j = Int64[]
	k = Int64[]
	vals2 = Float64[]
	
	sparse3tensor(i, j, k, vals2, nothing)
end