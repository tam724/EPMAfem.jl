# direct solver
const BackslashMatrix{T} = LazyOpMatrix{T, typeof(\), <:Tuple{<:MaterializedMatrix{T}}}
M(K::BackslashMatrix) = only(K.args)
A(K::BackslashMatrix) = A(M(K))

Base.size(K::BackslashMatrix) = size(A(K))
max_size(K::BackslashMatrix) = max_size(A(K))
isdiagonal(K::BackslashMatrix) = isdiagonal(A(K))

lazy_getindex(K::BackslashMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, K::BackslashMatrix, x::AbstractVector, α::Number, β::Number)
    A_, rem = materialize_with(ws, M(K))
    y .= α .* (A_ \ x) .+ β .* y
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:BackslashMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    A_, rem = materialize_with(ws, M(parent(Kt)))
    y .= α .* (transpose(A_) \ x) .+ β .* y
end

required_workspace(::typeof(mul_with!), K::BackslashMatrix, n, cache_notifier) = required_workspace(materialize_with, M(K), cache_notifier)

# krylov_minres
const KrylovMinresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.minres), <:Tuple{<:AbstractMatrix{T}}}
A(K::KrylovMinresMatrix) = K.args[1]

Base.size(K::KrylovMinresMatrix) = size(A(K))
max_size(K::KrylovMinresMatrix) = max_size(A(K))
isdiagonal(K::KrylovMinresMatrix) = isdiagonal(A(K))

lazy_getindex(K::KrylovMinresMatrix, i::Int, j::Int) = error("Cannot getindex")

# function krylov_minres_solver_view(temp, x::UnsafeArray)
#     n = size(x, 1)
#     T = eltype(x)
#     skt = solver.lin_solver_skeleton
#     stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
#     return MinresSolver{T, T, typeof(x)}(
#         n, 
#         n,
#         uview(T[], 1:0),
#         x,
#         uview(temp, 0*n+1:1*n), # r1
#         uview(temp, 1*n+1:2*n), # r2
#         uview(temp, 2*n+1:3*n), # w1
#         uview(temp, 3*n+1:4*n), # w2
#         uview(temp, 4*n+1:5*n), # y
#         uview(T[], 1:0), # v
#         skt.err_vec,
#         false,
#         stats)
# end

function mul_with!(ws::Workspace, y::AbstractVector, K::KrylovMinresMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(K), ws)
    CUDA.NVTX.@range "minres allocate" begin
        solver = Krylov.MinresSolver(A_, x) # this allocates!
    end
    CUDA.NVTX.@range "minres solve" begin
        Krylov.solve!(solver, A_, x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    end
    CUDA.NVTX.@range "minres copy" begin
        y .= α .* solver.x .+ β .* y
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:KrylovMinresMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(parent(Kt)), ws)
    CUDA.NVTX.@range "minres allocate" begin
        solver = Krylov.MinresSolver(A_, x) # this allocates!
    end
    CUDA.NVTX.@range "minres solve" begin
        Krylov.solve!(solver, transpose(A_), x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    end
    CUDA.NVTX.@range "minres copy" begin
        y .= α .* solver.x .+ β .* y
    end
end

function required_workspace(::typeof(mul_with!), K::KrylovMinresMatrix, n, cache_notifier)
    @assert n == 1
    return required_workspace(mul_with!, A(K), n, cache_notifier)
end

# krylov_gmres
const KrylovGmresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.gmres), <:Tuple{<:AbstractMatrix{T}}}
A(K::KrylovGmresMatrix) = K.args[1]

Base.size(K::KrylovGmresMatrix) = size(A(K))
max_size(K::KrylovGmresMatrix) = max_size(A(K))
isdiagonal(K::KrylovGmresMatrix) = isdiagonal(A(K))

lazy_getindex(K::KrylovGmresMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, K::KrylovGmresMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(K), ws)
    solver = Krylov.GmresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, A_, x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    y .= α .* solver.x .+ β .* y
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:KrylovGmresMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(parent(Kt)), ws)
    solver = Krylov.GmresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, transpose(A_), x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    y .= α .* solver.x .+ β .* y
end

function required_workspace(::typeof(mul_with!), K::KrylovGmresMatrix, n, cache_notifier)
    @assert n == 1
    return required_workspace(mul_with!, A(K), n, cache_notifier)
end

function schur_complement() end

const SchurMatrix{T} = LazyOpMatrix{T, typeof(schur_complement), <:NTuple{4, AbstractMatrix{T}}}
inv_AmBD⁻¹C(S::SchurMatrix) = S.args[1]
B(S::SchurMatrix) = S.args[2]
C(S::SchurMatrix) = S.args[3]
D⁻¹(S::SchurMatrix) = S.args[4]

# inner_solver(S::SchurMatrix) = S.op[2]
# inner_solver(St::Transpose{T, <:SchurMatrix{T}}) where T = parent(St).op[2]

function block_size(S::SchurMatrix)
    n1 = only_unique(size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(size(D⁻¹(S)))
    return (n1, n2)
end
block_size(St::Transpose{T, <:SchurMatrix{T}}) where T = block_size(parent(St))

function Base.size(S::SchurMatrix)
    n1, n2 = block_size(S)
    @assert size(B(S)) == (n1, n2)
    @assert size(C(S)) == (n2, n1)
    return duplicate(n1 + n2)
end
function max_size(S::SchurMatrix)
    n1 = only_unique(max_size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(max_size(D⁻¹(S)))
    @assert max_size(D(S)) == (n1, n2)
    @assert max_size(C(S)) == (n2, n1)
    return duplicate(n1 + n2)
end
isdiagonal(S::SchurMatrix) = false # should not happen..

lazy_getindex(S::SchurMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, S::SchurMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
    @assert α
    @assert !β

    n1, n2 = block_size(S)

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])

    x_ = @view(y[1:n1])
    y_ = @view(y[n1+1:n1+n2])

    mul_with!(ws, u, B(S)*D⁻¹(S), v, T(-1), true)
    mul_with!(ws, x_, inv_AmBD⁻¹C(S), u, true, false)
    mul_with!(ws, v, C(S), x_, T(-1), true)
    mul_with!(ws, y_, D⁻¹(S), v, true, false)
end

function mul_with!(ws::Workspace, y::AbstractVector, St::Transpose{T, <:SchurMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    @assert α
    @assert !β
    S = parent(St)

    n1, n2 = block_size(St)

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])

    x_ = @view(y[1:n1])
    y_ = @view(y[n1+1:n1+n2])

    mul_with!(ws, u, transpose(C(S)) * transpose(D⁻¹(S)), v, T(-1), true)
    mul_with!(ws, x_, transpose(inv_AmBD⁻¹C(S)), u, true, false)
    mul_with!(ws, v, transpose(B(S)), x_, T(-1), true)
    mul_with!(ws, y_, transpose(D⁻¹(S)), v, true, false)
end

function required_workspace(::typeof(mul_with!), S::SchurMatrix, n, cache_notifier)
    @assert n == 1
    maximum(A -> required_workspace(mul_with!, A, n, cache_notifier),
        (inv_AmBD⁻¹C(S), B(S)*D⁻¹(S), C(S), D⁻¹(S), transpose(C(S))*transpose(D⁻¹(S)), transpose(B(S))))
end

# this is a weird one.. (we implement the interface here..)
function half_schur_complement(BM::BlockMatrix, solver, fast_solver)
    A, B, C, D = blocks(BM)
    D⁻¹ = fast_solver(D)
    inv_AmBD⁻¹C = solver(A - B * D⁻¹ * C)
    return lazy(half_schur_complement, inv_AmBD⁻¹C, B, C, D⁻¹)
end

const HalfSchurMatrix{T} = LazyOpMatrix{T,  typeof(half_schur_complement), <:NTuple{4, AbstractMatrix{T}}}
inv_AmBD⁻¹C(S::HalfSchurMatrix) = S.args[1]
B(S::HalfSchurMatrix) = S.args[2]
C(S::HalfSchurMatrix) = S.args[3]
D⁻¹(S::HalfSchurMatrix) = S.args[4]

function block_size(S::HalfSchurMatrix)
    n1 = only_unique(size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(size(D⁻¹(S)))
    return (n1, n2)
end
block_size(St::Transpose{T, <:HalfSchurMatrix{T}}) where T = block_size(parent(St))

function Base.size(S::Union{HalfSchurMatrix, Transpose{T, <:HalfSchurMatrix{T}}}) where T
    n1, n2 = block_size(S)
    return (n1, n1 + n2)
end
function max_size(S::Union{HalfSchurMatrix, Transpose{T, <:HalfSchurMatrix{T}}}) where T
    n1 = only_unique(max_size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(max_size(D⁻¹(S)))
    @assert max_size(D(S)) == (n1, n2)
    @assert max_size(C(S)) == (n2, n1)
    return (n1, n1 + n2)
end
isdiagonal(S::HalfSchurMatrix) = false # not even square :D

lazy_getindex(S::HalfSchurMatrix, i::Int, j::Int) = error("Cannot getindex")
function LinearAlgebra.transpose(S::HalfSchurMatrix{T}) where T
    @warn "this behaves weird!"
    return Transpose(S)
end

function mul_with!(ws::Workspace, y::AbstractVector, S::HalfSchurMatrix{T}, x::AbstractVector, α::Number, β::Number) where T
    @assert α
    @assert !β

    n1, n2 = block_size(S)

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])
    x_ = @view(y[1:n1])

    mul_with!(ws, u, B(S)*D⁻¹(S), v, T(-1), true)
    mul_with!(ws, x_, inv_AmBD⁻¹C(S), u, true, false)
end

function mul_with!(ws::Workspace, y::AbstractVector, St::Transpose{T, <:HalfSchurMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    @assert α
    @assert !β
    S = parent(St)

    n1, n2 = block_size(St)

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])
    x_ = @view(y[1:n1])

    mul_with!(ws, u, transpose(C(S))*transpose(D⁻¹(S)), v, T(-1), true)
    mul_with!(ws, x_, transpose(inv_AmBD⁻¹C(S)), u, true, false)
end

function required_workspace(::typeof(mul_with!), S::HalfSchurMatrix, n, cache_notifier)
    @assert n == 1
    maximum(A -> required_workspace(mul_with!, A, n, cache_notifier), (inv_AmBD⁻¹C(S), B(S)*D⁻¹(S), transpose(C(S))*transpose(D⁻¹(S))))
end
