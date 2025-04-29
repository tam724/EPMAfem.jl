Base.@kwdef @concrete struct DiscreteImplicitMidpointPNSystem <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    A
    rhs

    lin_solver
end

function implicit_midpoint(pbl::DiscretePNProblem, solver; solver_kwargs...)
    nb = EPMAfem.n_basis(pbl)
    ns = EPMAfem.n_sums(pbl)

    arch = architecture(pbl)
    T = base_type(arch)

    cache = allocate_vec(arch, max(nb.nx.p, nb.nx.m)*max(nb.nΩ.p, nb.nΩ.m))
    cache2 = allocate_vec(arch, max(nb.nΩ.p, nb.nΩ.m))

    coeffs = (a = Vector{T}(undef, ns.ne), c = [Vector{T}(undef, ns.nσ) for _ in 1:ns.ne])

    ρp, ρm, ∂p, ∇pm = space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = direction_matrices(pbl)

    A = ZMatrix2{T}(ρp, Ip, kp, coeffs.a, coeffs.c, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.ne, ns.nσ, mat_view(cache, nb.nx.p, nb.nΩ.p), Diagonal(@view(cache2[1:nb.nΩ.p])))
    B = DMatrix2{T}(∇pm, Ωpm, nb.nx.p, nb.nx.m, nb.nΩ.m, nb.nΩ.p, ns.nd, mat_view(cache, nb.nx.p, nb.nΩ.m))
    C = ZMatrix2{T}(ρm, Im, km, coeffs.a, coeffs.c, nb.nx.m, nb.nx.m, nb.nΩ.m, nb.nΩ.m, ns.ne, ns.nσ, mat_view(cache, nb.nx.m, nb.nΩ.m), Diagonal(@view(cache2[1:nb.nΩ.m])))
    D = DMatrix2{T}(∂p, absΩp, nb.nx.p, nb.nx.p, nb.nΩ.p, nb.nΩ.p, ns.nd, mat_view(cache, nb.nx.p, nb.nΩ.p))

    Δϵ = step(energy_model(pbl.model))
    BM = BlockMat2{T}(A, B, C, D, nb.nx.p*nb.nΩ.p, nb.nx.m*nb.nΩ.m, Ref(0.5), Ref(-0.5), Ref(0.5), Ref(Δϵ), Ref(false))
    lin_solver = solver(vec_type(arch), BM; solver_kwargs...)
    BM.sym[] = symmetrize_blockmat(lin_solver)

    n = nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m
    rhs = allocate_vec(arch, n)

    return DiscreteImplicitMidpointPNSystem(
        problem = pbl,

        coeffs = coeffs,
        A = BM,
        rhs = rhs,

        lin_solver = lin_solver
    )
end

function Base.adjoint(A::DiscreteImplicitMidpointPNSystem)
    return DiscreteImplicitMidpointPNSystem(adjoint=!A.adjoint, problem=A.problem, coeffs=A.coeffs, A=A.A, rhs=A.rhs, lin_solver=A.lin_solver)
end

function step_nonadjoint!(x, system::DiscreteImplicitMidpointPNSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, system.A.sym[])
    problem = system.problem
    # ip1 = i
    # i = i-1
    ns = n_sums(system.problem)
    for e in 1:ns.ne
        si, sip1 = problem.s[e, minus1(idx)], problem.s[e, idx]
        τi, τip1 = problem.τ[e, minus1(idx)], problem.τ[e, idx]
        system.coeffs.a[e] = -sip1 / Δϵ + 0.5 * τip1
        for sc in 1:ns.nσ
            σi, σip1 = problem.σ[e, sc, minus1(idx)],  problem.σ[e, sc, idx]
            system.coeffs.c[e][sc] = -0.5 * σip1
        end
    end
    system.A.Δ[] = Δϵ
    # minus because we have to bring b to the right side of the equation
    mul!(system.rhs, system.A, x, -1.0, true)

    #update the matrix
    for e in 1:ns.ne
        si, sip1 = problem.s[e, minus1(idx)], problem.s[e, idx]
        τi, τip1 = problem.τ[e, minus1(idx)], problem.τ[e, idx]
        system.coeffs.a[e] = si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi, σip1 = problem.σ[e, sc, minus1(idx)],  problem.σ[e, sc, idx]
            system.coeffs.c[e][sc] = -0.5 * σi
        end
    end

    pn_linsolve!(system.lin_solver, x, system.A, system.rhs)

    # solver = solver_from_buf(x, system.lin_solver)
    # Krylov.solve!(solver, system.A, system.rhs, rtol=system.rtol, atol=system.atol)
end

function step_adjoint!(x, system::DiscreteImplicitMidpointPNSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if !system.adjoint @warn "Trying to step_adjoint with system marked as nonadjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(pnsystem.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, plus½(idx), -Δϵ, true)
    problem = system.problem
    # ip12 = i+1
    # im12 = i
    ns = n_sums(system.problem)
    for e in 1:ns.ne
        si = problem.s[e, plus½(idx)]
        τi = problem.τ[e, plus½(idx)]
        system.coeffs.a[e] = -si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi = problem.σ[e, sc, plus½(idx)]
            system.coeffs.c[e][sc] = -0.5 * σi
        end
    end
    system.A.Δ[] = Δϵ
    # minus because we have to bring b to the right side of the equation
    mul!(system.rhs, transpose(system.A), x, -1.0, true)

    #update the matrix
    for e in 1:ns.ne
        si = problem.s[e, plus½(idx)]
        τi = problem.τ[e, plus½(idx)]
        system.coeffs.a[e] = si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi = problem.σ[e, sc, plus½(idx)]
            system.coeffs.c[e][sc] = -0.5 * σi
        end
    end

    pn_linsolve!(system.lin_solver, x, transpose(system.A), system.rhs)

    # solver = solver_from_buf(x, system.lin_solver)
    # Krylov.solve!(solver, transpose(system.A), system.rhs, rtol=system.rtol, atol=system.atol)
end

function allocate_solution_vector(system::DiscreteImplicitMidpointPNSystem)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end
