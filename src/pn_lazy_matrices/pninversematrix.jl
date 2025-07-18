# direct solver
const BackslashMatrix{T} = LazyOpMatrix{T, typeof(\), <:Tuple{<:AbstractMatrix{T}}}
A(K::BackslashMatrix) = only(K.args)

Base.size(K::BackslashMatrix) = size(A(K))
max_size(K::BackslashMatrix) = max_size(A(K))
isdiagonal(K::BackslashMatrix) = isdiagonal(A(K))

lazy_getindex(K::BackslashMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, K::BackslashMatrix, x::AbstractVector, α::Number, β::Number)
    A_, rem = materialize_with(ws, materialize(A(K)))
    y .= α .* (A_ \ x) .+ β .* y
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:BackslashMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    A_, rem = materialize_with(ws, materialize(A(parent(Kt))))
    y .= α .* (transpose(A_) \ x) .+ β .* y
end

required_workspace(::typeof(mul_with!), K::BackslashMatrix) = required_workspace(materialize_with, materialize(A(K)))

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
    solver = Krylov.MinresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, A_, x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    y .= α .* solver.x .+ β .* y
end

function mul_with!(ws::Workspace, y::AbstractVector, Kt::Transpose{T, <:KrylovMinresMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(parent(Kt)), ws)
    solver = Krylov.MinresSolver(A_, x) # this allocates!
    Krylov.solve!(solver, transpose(A_), x; rtol=T(sqrt(eps(Float64))), atol=zero(T))
    y .= α .* solver.x .+ β .* y
end

required_workspace(::typeof(mul_with!), K::KrylovMinresMatrix) = required_workspace(mul_with!, A(K))

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

required_workspace(::typeof(mul_with!), K::KrylovGmresMatrix) = required_workspace(mul_with!, A(K))


function schur_complement() end

const SchurInnerSolver = Any # Union{typeof(Krylov.minres), typeof(Krylov.gmres), typeof(\), typeof(Krylov.minres ∘ cache), typeof(Krylov.gmres ∘ cache), typeof(\)}
const SchurMatrix{T} = LazyOpMatrix{T, <:Tuple{typeof(schur_complement), <:SchurInnerSolver}, <:Tuple{Union{BlockMatrix{T}, Transpose{T, <:BlockMatrix{T}}}}}
BM(S::SchurMatrix) = only(S.args)
BM(St::Transpose{T, <:SchurMatrix{T}}) where T = transpose(BM(parent(St)))
inner_solver(S::SchurMatrix) = S.op[2]
inner_solver(St::Transpose{T, <:SchurMatrix{T}}) where T = parent(St).op[2]

Base.size(S::SchurMatrix) = size(BM(S))
max_size(S::SchurMatrix) = max_size(BM(S))
isdiagonal(S::SchurMatrix) = isdiagonal(BM(S))

lazy_getindex(S::SchurMatrix, i::Int, j::Int) = error("Cannot getindex")

_schur_blocks(S::SchurMatrix) = lazy.(blocks(BM(S)))

function _schur_components(S::Union{SchurMatrix, Transpose{T, <:SchurMatrix{T}}}) where T
    A, B, C, D = lazy.(blocks(BM(S)))
    D⁻¹ = cache(inv!(D))
    return B * D⁻¹, inner_solver(S)(A - B * D⁻¹ * C), unwrap(C), D⁻¹
end

function mul_with!(ws::Workspace, y::AbstractVector, S::Union{SchurMatrix, Transpose{T, <:SchurMatrix{T}}}, x::AbstractVector, α::Number, β::Number) where T
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

# function mul_with!(ws::Workspace, y::AbstractVector, St::Transpose{T, <:SchurMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
#     @assert α
#     @assert !β

#     n1, n2 = block_size(BM(parent(St)))

#     u = @view(x[1:n1])
#     v = @view(x[n1+1:n1+n2])

#     x_ = @view(y[1:n1])
#     y_ = @view(y[n1+1:n1+n2])

#     t_BD⁻¹, t_inv_AmBD⁻¹C, t_C, t_D⁻¹ = _schur_components(St)

#     mul_with!(ws, u, t_BD⁻¹, v, -1, true)
#     mul_with!(ws, x_, t_inv_AmBD⁻¹C, u, true, false)
#     mul_with!(ws, v, t_C, x_, -1, true)
#     mul_with!(ws, y_, t_D⁻¹, v, true, false)
# end


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
isdiagonal(S::HalfSchurMatrix) = false # not even square :D

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
