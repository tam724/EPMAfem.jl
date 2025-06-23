@concrete struct DiscreteMonochromPNProblem
    model
    arch

    # coefficients
    τ
    σ

    # discretizations
    space_discretization
    direction_discretization
end

architecture(problem::DiscreteMonochromPNProblem) = problem.arch
n_basis(problem::DiscreteMonochromPNProblem) = n_basis(problem.model)
n_sums(problem::DiscreteMonochromPNProblem) = (nd = length(dimensions(problem.model)), nσ = 1, ne = size(problem.τ, 1))

Base.show(io::IO, p::DiscreteMonochromPNProblem) = print(io, "MonochromPNProblem [$(n_basis(p)) and $(n_sums(p))]")
Base.show(io::IO, ::MIME"text/plain", p::DiscreteMonochromPNProblem) = show(io, p)

Base.@kwdef @concrete struct DiscreteMonochromPNSystem
    adjoint::Bool=false
    problem

    coeffs
    A
    rhs

    lin_solver
end

function system(pbl::DiscreteMonochromPNProblem, solver; solver_kwargs...)
    nb = n_basis(pbl)
    ns = n_sums(pbl)

    arch = architecture(pbl)
    T = base_type(arch)

    cache = allocate_vec(arch, max(nb.nx.p, nb.nx.m)*max(nb.nΩ.p, nb.nΩ.m))
    cache2 = allocate_vec(arch, max(nb.nΩ.p, nb.nΩ.m))

    coeffs = (a = [T(pbl.τ[i]) for i in 1:ns.ne], c = [[T(pbl.σ[i])] for i in 1:ns.ne])

    A = ZMatrix2{T}(pbl.space_discretization.ρp, pbl.direction_discretization.Ip, pbl.direction_discretization.kp, coeffs.a, coeffs.c, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.ne, ns.nσ, mat_view(cache, nb.nx.p, nb.nΩ.p), Diagonal(@view(cache2[1:nb.nΩ.p])))
    B = DMatrix2{T}(pbl.space_discretization.∇pm, pbl.direction_discretization.Ωpm, nb.nx.p, nb.nx.m, nb.nΩ.m, nb.nΩ.p, ns.nd, mat_view(cache, nb.nx.p, nb.nΩ.m))
    C = ZMatrix2{T}(pbl.space_discretization.ρm, pbl.direction_discretization.Im, pbl.direction_discretization.km, coeffs.a, coeffs.c, nb.nx.m, nb.nx.m, nb.nΩ.m, nb.nΩ.m, ns.ne, ns.nσ, mat_view(cache, nb.nx.m, nb.nΩ.m), Diagonal(@view(cache2[1:nb.nΩ.m])))
    D = DMatrix2{T}(pbl.space_discretization.∂p, pbl.direction_discretization.absΩp, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.nd, mat_view(cache, nb.nx.p, nb.nΩ.p))

    BM = BlockMat2{T}(A, B, C, D, nb.nx.p*nb.nΩ.p, nb.nx.m*nb.nΩ.m, Ref(1.0), Ref(-1.0), Ref(1.0), Ref(1.0), Ref(false))
    lin_solver = solver(vec_type(arch), BM; solver_kwargs...)
    BM.sym[] = symmetrize_blockmat(lin_solver)

    n = nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m
    rhs = allocate_vec(arch, n)
    return DiscreteMonochromPNSystem(
        problem=pbl,

        coeffs = coeffs,
        A = BM,
        rhs = rhs,
        lin_solver = lin_solver
    )
end

function solve(x, system::DiscreteMonochromPNSystem, B::Vector)
    assemble!(system.rhs, B, -1, system.A.sym[])
    normalize!(system.rhs)
    pn_linsolve!(system.lin_solver, x, system.A, system.rhs)
end

function allocate_solution_vector(system::DiscreteMonochromPNSystem)
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end
