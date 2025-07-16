# MINRES

@concrete struct WrappedKrylovMinres <: AbstractPNSystemSolver
    arch
    lin_solver_skeleton
end

function solver_view(solver::WrappedKrylovMinres, x::UnsafeArray)
    n = size(x, 1)
    T = base_type(solver.arch)
    skt = solver.lin_solver_skeleton
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T, T, typeof(x)}(
        n, 
        n,
        uview(skt.Δx, 1:0),
        x,
        uview(skt.r1, 1:n),
        uview(skt.r2, 1:n),
        uview(skt.w1, 1:n),
        uview(skt.w2, 1:n),
        uview(skt.y, 1:n),
        uview(skt.v, 1:0),
        skt.err_vec,
        false,
        stats)
end


function allocate_solver(::typeof(Krylov.minres), n, arch)
    # A should be symmetric here!
    # lin_solver = MinresSolver(n, n, vec_type(arch))
    lin_solver_skeleton = (
        # m = lin_solver.m,
        # n = lin_solver.n,
        Δx = allocate_vec(arch, 0),
        # x is not part of the skeleton
        r1 = allocate_vec(arch, n),
        r2 = allocate_vec(arch, n),
        w1 = allocate_vec(arch, n),
        w2 = allocate_vec(arch, n),
        y = allocate_vec(arch, n),
        v = allocate_vec(arch, 0),
        err_vec = zeros(base_type(arch), 5)
    )
    return WrappedKrylovMinres(arch, lin_solver_skeleton)
end

function linsolve!(solver::WrappedKrylovMinres, x, A, b)
    krylov_solver = solver_view(solver, cuview(x, :))
    Krylov.minres!(krylov_solver, A, cuview(b, :))
end

# GMRES

@concrete struct WrappedKrylovGmres <: AbstractPNSystemSolver end

allocate_solver(::typeof(Krylov.gmres), n, arch) = WrappedKrylovGmres()

function linsolve!(::WrappedKrylovGmres, x, A, b)
    x_, stats = Krylov.gmres(A, b)
    x .= x_
end

# DIRECT
@concrete struct WrappedDirectSolver <: AbstractPNSystemSolver end

allocate_solver(::typeof(\), n, arch) = WrappedDirectSolver()

function linsolve!(::WrappedDirectSolver, x, A, b)
   A_ = assemble_from_op(A)
    x .= A_ \ b
end
