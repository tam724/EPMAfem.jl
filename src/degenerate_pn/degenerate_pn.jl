@concrete struct DiscreteDegeneratePNProblem
    model
    arch

    # direction
    Ω

    # coefficients
    τ

    # discretizations
    space_discretization
end

architecture(problem::DiscreteDegeneratePNProblem) = problem.arch
function n_basis(problem::DiscreteDegeneratePNProblem)
    nx = SpaceModels.n_basis(problem.model)
    nΩ = (p=1, m=1)
    return (nx=nx, nΩ=nΩ)
end

n_sums(problem::DiscreteDegeneratePNProblem) = (nd = length(Dimensions.dimensions(SpaceModels.dimensionality(problem.model))), nσ = 1, ne = size(problem.τ, 1))


Base.@kwdef @concrete struct DiscreteDegeneratePNSystem
    adjoint::Bool=false
    problem

    coeffs
    A
    rhs

    lin_solver
end

function system(pbl::DiscreteDegeneratePNProblem, solver; solver_kwargs...)
    nb = n_basis(pbl)
    ns = n_sums(pbl)

    arch = architecture(pbl)
    T = base_type(arch)

    coeffs = (τ = [Diagonal(allocate_vec(arch, 1)) for _ in 1:ns.ne], Ωpm = [allocate_mat(arch, 1, 1) for _ in 1:ns.nd], absΩp = [allocate_mat(arch, 1, 1) for _ in 1:ns.nd])
    for e in 1:ns.ne
        coeffs.τ[e].diag[:] .= pbl.τ[e]
    end
    for d in 1:ns.nd
        coeffs.Ωpm[d][:] .= pbl.Ω[d]
        coeffs.absΩp[d][:] .= abs(pbl.Ω[d])
    end
    cache = allocate_vec(arch, max(nb.nx.p, nb.nx.m))

    A = DMatrix2{T}(pbl.space_discretization.ρp, coeffs.τ, nb.nx.p, nb.nx.p, 1, 1, ns.ne, mat_view(cache, nb.nx.p, 1))
    B = DMatrix2{T}(pbl.space_discretization.∇pm, coeffs.Ωpm, nb.nx.p, nb.nx.m, 1, 1, ns.nd, mat_view(cache, nb.nx.p, 1))
    C = DMatrix2{T}(pbl.space_discretization.ρm, coeffs.τ, nb.nx.m, nb.nx.m, 1, 1, ns.ne, mat_view(cache, nb.nx.m, 1))
    D = DMatrix2{T}(pbl.space_discretization.∂p, coeffs.absΩp, nb.nx.p, nb.nx.p, 1, 1, ns.nd, mat_view(cache, nb.nx.p, 1))

    BM = BlockMat2{T}(A, B, C, D, nb.nx.p, nb.nx.m, Ref(1.0), Ref(-1.0), Ref(1.0), Ref(1.0), Ref(false))
    lin_solver = solver(vec_type(arch), BM; solver_kwargs...)
    BM.sym[] = symmetrize_blockmat(lin_solver)
    
    n = nb.nx.p + nb.nx.m
    rhs = allocate_vec(arch, n)
    return DiscreteDegeneratePNSystem(
        problem=pbl,

        coeffs = coeffs,
        A = BM,
        rhs = rhs,
        lin_solver = lin_solver
    )
end

function solve(x, system::DiscreteDegeneratePNSystem, (bp, bm))
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    system.rhs[1:nxp] .= bp./(-sqrt((2)))
    if system.A.sym[]
        system.rhs[nxp+1:nxp+nxm] .= bm./(sqrt((2)))
    else
        system.rhs[nxp+1:nxp+nxm] .= bm./(-sqrt(2))
    end
    # assemble!(system.rhs, b, -1, system.A.sym[])
    pn_linsolve!(system.lin_solver, x, system.A, system.rhs)
end

function allocate_solution_vector(system::DiscreteDegeneratePNSystem)
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end

include("degenerate_pnequations.jl")
include("degenerate_pndiscretization.jl")
