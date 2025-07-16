# direct solver
const BackslashMatrix{T} = LazyOpMatrix{T, typeof(\), <:Tuple{<:AbstractMatrix{T}}}
A(K::BackslashMatrix) = only(K.args)

Base.size(K::BackslashMatrix) = size(A(K))
max_size(K::BackslashMatrix) = max_size(A(K))

isdiagonal(K::BackslashMatrix) = isdiagonal(A(K))
LinearAlgebra.transpose(K::BackslashMatrix) = lazy(\, transpose(A(K)))

lazy_getindex(K::BackslashMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, K::BackslashMatrix, x::AbstractVector, α::Number, β::Number)
    A_, rem = materialize_with(ws, materialize(A(K)), nothing)
    y .= α .* (A_ \ x) .+ β .* y
end
required_workspace(::typeof(mul_with!), K::BackslashMatrix) = required_workspace(materialize_with, materialize(A(K)))

# krylov_minres
const KrylovMinresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.minres), <:Tuple{<:AbstractMatrix{T}}}
A(K::KrylovMinresMatrix) = K.args[1]

Base.size(K::KrylovMinresMatrix) = size(A(K))
max_size(K::KrylovMinresMatrix) = max_size(A(K))

isdiagonal(K::KrylovMinresMatrix) = isdiagonal(A(K))
LinearAlgebra.transpose(K::KrylovMinresMatrix) = lazy(Krylov.minres, transpose(A(K)))

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

function mul_with!(ws::Workspace, y::AbstractVector, K::KrylovMinresMatrix, x::AbstractVector, α::Number, β::Number)
    A_ = NotSoLazy{eltype(K)}(A(K), ws)
    solver = Krylov.MinresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, A_, x)
    y .= α .* solver.x .+ β .* y
end

required_workspace(::typeof(mul_with!), K::KrylovMinresMatrix) = required_workspace(mul_with!, A(K))

# krylov_gmres
const KrylovGmresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.gmres), <:Tuple{<:AbstractMatrix{T}}}
A(K::KrylovGmresMatrix) = K.args[1]

Base.size(K::KrylovGmresMatrix) = size(A(K))
max_size(K::KrylovGmresMatrix) = max_size(A(K))

isdiagonal(K::KrylovGmresMatrix) = isdiagonal(A(K))
LinearAlgebra.transpose(K::KrylovGmresMatrix) = lazy(Krylov.gmres, transpose(A(K)))

lazy_getindex(K::KrylovGmresMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, K::KrylovGmresMatrix, x::AbstractVector, α::Number, β::Number)
    A_ = NotSoLazy{eltype(K)}(A(K), ws)
    solver = Krylov.GmresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, A_, x)
    y .= α .* solver.x .+ β .* y
end

required_workspace(::typeof(mul_with!), K::KrylovGmresMatrix) = required_workspace(mul_with!, A(K))


function schur_complement() end

const SchurInnerSolver = Union{typeof(Krylov.minres), typeof(Krylov.gmres), typeof(\)}
const SchurMatrix{T} = LazyOpMatrix{T, <:Tuple{typeof(schur_complement), <:SchurInnerSolver}, <:Tuple{Union{BlockMatrix{T}, Transpose{T, <:BlockMatrix{T}}}}}
BM(S::SchurMatrix) = only(S.args)
inner_solver(S::SchurMatrix) = S.op[2]

Base.size(S::SchurMatrix) = size(BM(S))
max_size(S::SchurMatrix) = max_size(BM(S))
LinearAlgebra.transpose(S::SchurMatrix) = lazy(S.op, transpose(BM(S)))

lazy_getindex(S::SchurMatrix, i::Int, j::Int) = error("Cannot getindex")

function _schur_components(S)
    A, B, C, D = lazy.(blocks(BM(S)))
    D⁻¹ = cache(inv!(D))
    return B * D⁻¹, inner_solver(S)(A - B * D⁻¹ * C), unwrap(C), D⁻¹
end

function mul_with!(ws::Workspace, y::AbstractVector, S::SchurMatrix, x::AbstractVector, α::Number, β::Number)
    @assert α
    @assert !β

    n1, n2 = block_size(BM(S))

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])

    x_ = @view(y[1:n1])
    y_ = @view(y[n1+1:n1+n2])

    BD⁻¹, inv_AmBD⁻¹C, C, D⁻¹ = _schur_components(S)

    mul_with!(ws, u, BD⁻¹, v, -1, true)
    mul_with!(ws, x_, inv_AmBD⁻¹C, u, true, false)
    mul_with!(ws, v, C, x_, -1, true)
    mul_with!(ws, y_, D⁻¹, v, true, false)
end

function required_workspace(::typeof(mul_with!), S::SchurMatrix)
    BD⁻¹, inv_AmBD⁻¹C, C, D⁻¹ = _schur_components(S)
    ws = required_workspace(mul_with!, BD⁻¹)
    ws = max(ws, required_workspace(mul_with!, inv_AmBD⁻¹C))
    ws = max(ws, required_workspace(mul_with!, C))
    ws = max(ws, required_workspace(mul_with!, D⁻¹))
    return ws
end

# this is a weird one..
function half_schur_complement() end
const HalfSchurMatrix{T} = LazyOpMatrix{T, <:Tuple{typeof(half_schur_complement), <:SchurInnerSolver}, <:Tuple{Union{BlockMatrix{T}, Transpose{T, <:BlockMatrix{T}}}}}
BM(S::HalfSchurMatrix) = only(S.args)
inner_solver(S::HalfSchurMatrix) = S.op[2]

function Base.size(S::HalfSchurMatrix)
    n1, n2 = block_size(BM(S))
    return (n1, n1 + n2)
end
function max_size(S::HalfSchurMatrix)
    n1, n2 = max_block_size(BM(S))
    return (n1, n1 + n2)
end

LinearAlgebra.transpose(S::HalfSchurMatrix) = lazy(S.op, transpose(BM(S)))

lazy_getindex(S::HalfSchurMatrix, i::Int, j::Int) = error("Cannot getindex")

function _half_schur_components(S)
    A, B, C, D = lazy.(blocks(BM(S)))
    D⁻¹ = cache(inv!(D))
    return B * D⁻¹, inner_solver(S)(A - B * D⁻¹ * C)
end

function mul_with!(ws::Workspace, y::AbstractVector, S::HalfSchurMatrix, x::AbstractVector, α::Number, β::Number)
    @assert α
    @assert !β

    n1, n2 = block_size(BM(S))

    u = @view(x[1:n1])
    v = @view(x[n1+1:n1+n2])
    x_ = @view(y[1:n1])

    BD⁻¹, inv_AmBD⁻¹C = _half_schur_components(S)

    mul_with!(ws, u, BD⁻¹, v, -1, true)
    mul_with!(ws, x_, inv_AmBD⁻¹C, u, true, false)
end

function required_workspace(::typeof(mul_with!), S::HalfSchurMatrix)
    BD⁻¹, inv_AmBD⁻¹C = _half_schur_components(S)
    ws = required_workspace(mul_with!, BD⁻¹)
    ws = max(ws, required_workspace(mul_with!, inv_AmBD⁻¹C))
    return ws
end
