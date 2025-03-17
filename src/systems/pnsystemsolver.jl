abstract type AbstractPNSystemSolver end

# cuview(A::Array, slice) = uview(A, slice)
# cuview(A::CuArray, slice) = view(A, slice)

@concrete struct PNSchurSolver <: AbstractPNSystemSolver
    C_ass
    cache

    b_schur
    lin_solver
end

symmetrize_blockmat(::PNSchurSolver) = false

function PNSchurSolver(VT, A::BlockMat2{T}; solver=PNKrylovMinresSolver, solver_kwargs...) where T
    @assert isdiag(A.C)
    n_C = size(A.C)[1]

    # allocate cache for the diagonal
    C_ass = Diagonal(VT(undef, n_C))
    cache = VT(undef, n_C)
    # N = SchurBlockMat2{T}(A, C_ass, cache)
    b_schur = VT(undef, size(A.A)[1])

    lin_solver = solver(VT, size(A.A); solver_kwargs...)
    PNSchurSolver(C_ass, cache, b_schur, lin_solver)
end

function pn_linsolve!(S::PNSchurSolver, x, A, b)

    # solver = PNDirectSolver()
    # x_ = zeros(size(A)[2])
    # pn_linsolve!(solver, x_, A, b)

    N = SchurBlockMat2{eltype(A)}(A, S.C_ass, S.cache)

    update_cache!(N)

    n1, _ = block_size(A)

    x1 = @view(x[1:n1])

    schur_rhs!(S.b_schur, b, N)
    pn_linsolve!(S.lin_solver, x1, N, S.b_schur)
    schur_sol!(x, b, N)

    # @show maximum(abs.(x_ .- x))
end

## DIRECT SOLVER

@concrete struct PNDirectSolver <: AbstractPNSystemSolver
end

symmetrize_blockmat(::PNDirectSolver) = false


function PNDirectSolver(_, _)
    return PNDirectSolver()
end

function pn_linsolve!(::PNDirectSolver, x, A, b)
    A_ass = assemble_from_op(A)
    x .= A_ass \ b
end

## Krylov MINRES Solver

@concrete struct PNKrylovMinresSolver <: AbstractPNSystemSolver
    atol
    rtol

    # krylov stuff
    m
    n
    VT
    Δx
    r1
    r2
    w1
    w2
    y
    v
    err_vec
end

symmetrize_blockmat(::PNKrylovMinresSolver) = true

function PNKrylovMinresSolver(VT, (m, n)::Tuple{<:Int, <:Int}; tol=nothing, window=5)
    @assert m == n
    T = eltype(VT)

    atol = T(0)
    if isnothing(tol)
        rtol = T(sqrt(eps(Float64)))
    else
        rtol = T(tol)
    end

    return PNKrylovMinresSolver(
        atol,
        rtol,

        m,
        n,
        VT,
        VT(undef, 0),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, 0),
        zeros(T, window))
end

function PNKrylovMinresSolver(VT, A::BlockMat2; tol=nothing, window=5)
    m, n = size(A)
    @assert m == n
    T = eltype(VT)

    atol = T(0)
    if isnothing(tol)
        rtol = T(sqrt(eps(Float64)))
    else
        rtol = T(tol)
    end

    return PNKrylovMinresSolver(
        atol,
        rtol,

        m,
        n,
        VT,
        VT(undef, 0),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, 0),
        zeros(T, window))
end

function get_solver(x, S::PNKrylovMinresSolver)
    @assert x isa S.VT
    T = eltype(S.VT)
    @assert all(sz -> sz == S.n, (length(x), length(S.r1), length(S.r2), length(S.w1), length(S.w2), length(S.y))) 
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T,T,S.VT}(S.m, S.n, S.Δx, x, S.r1, S.r2, S.w1, S.w2, S.y, S.v, S.err_vec, false, stats)
end

function get_solver(x::UnsafeArray, S::PNKrylovMinresSolver)
    @assert eltype(x) == eltype(S.VT)
    T = eltype(S.VT)
    @assert all(sz -> sz == S.n, (length(x), length(S.r1), length(S.r2), length(S.w1), length(S.w2), length(S.y))) 
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T,T,UnsafeArray{T, 1}}(S.m, S.n, uview(S.Δx, 1:0), x, uview(S.r1, 1:S.n), uview(S.r2, 1:S.n), uview(S.w1, 1:S.n), uview(S.w2, 1:S.n), uview(S.y, 1:S.n), uview(S.v, 1:0), S.err_vec, false, stats)
end

function pn_linsolve!(S::PNKrylovMinresSolver, x, A, b)
    solver = get_solver(cuview(x, :), S)
    Krylov.solve!(solver, A, cuview(b, :), rtol=S.rtol, atol=S.atol)
end