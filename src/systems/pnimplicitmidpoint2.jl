Base.@kwdef @concrete struct DiscretePNSystem2 <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    BM
    BM⁻¹
    rhs
end

function build_coeffs_and_mat_blocks(pbl::DiscretePNProblem)
    T = base_type(architecture(pbl))

    ns = EPMAfem.n_sums(pbl)

    Δϵ = step(energy_model(pbl.model))

    coeffs = (a = [Ref(zero(T)) for _ in 1:ns.ne], c = [[Ref(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne], Δ=Ref(Δϵ), γ=Ref(0.5), δ=Ref(-0.5), δt=Ref(0.5))
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(pbl)

    A = sum(kron_AXB(ρp[i], coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ)) for i in 1:ns.ne)
    B = sum(kron_AXB(∇pm[i], Ωpm[i]) for i in 1:ns.nd)
    C = sum(kron_AXB(ρm[i], coeffs.a[i]*Im + sum(coeffs.c[i][j]*km[i][j] for j in 1:ns.nσ)) for i in 1:ns.ne)
    D = sum(kron_AXB(∂p[i], absΩp[i]) for i in 1:ns.nd)

    UL = coeffs.Δ*(A + coeffs.γ*D)
    UR = coeffs.Δ*(coeffs.δ*B)
    LL = T(-1)*(coeffs.Δ*(coeffs.δt*transpose(B)))
    LR = T(-1)*(coeffs.Δ*C) 

    return coeffs, UL, UR, LL, LR
end

function implicit_midpoint2(pbl::DiscretePNProblem, solver::Union{typeof(Krylov.minres), typeof(Krylov.gmres), })
    arch = architecture(pbl)

    coeffs, UL, UR, LL, LR = build_coeffs_and_mat_blocks(pbl)

    lazy_BM = [
        UL UR
        LL LR
        ]

    lazy_BM⁻¹ = lazy(solver, lazy_BM)

    BM, BM⁻¹ = unlazy((lazy_BM, lazy_BM⁻¹), vec_size -> allocate_vec(arch, vec_size))
    rhs = allocate_vec(arch, size(BM, 1))

    return DiscretePNSystem2(
        problem = pbl,
        coeffs = coeffs,
        BM = BM,
        BM⁻¹ = BM⁻¹,
        rhs = rhs,
    )
end

function implicit_midpoint_coeffs_nonadjoint_rhs!(coeffs, problem, idx, Δϵ)
    # ip1 = i
    # i = i-1
    ns = n_sums(problem)
    for e in 1:ns.ne
        si, sip1 = problem.s[e, minus1(idx)], problem.s[e, idx]
        τi, τip1 = problem.τ[e, minus1(idx)], problem.τ[e, idx]
        coeffs.a[e][] = -sip1 / Δϵ + 0.5 * τip1
        for sc in 1:ns.nσ
            σi, σip1 = problem.σ[e, sc, minus1(idx)],  problem.σ[e, sc, idx]
            coeffs.c[e][sc][] = -0.5 * σip1
        end
    end
end

function implicit_midpoint_coeffs_adjoint_rhs!(coeffs, problem, idx, Δϵ)
    # ip12 = i+1
    # im12 = i
    ns = n_sums(problem)
    for e in 1:ns.ne
        si = problem.s[e, plus½(idx)]
        τi = problem.τ[e, plus½(idx)]
        coeffs.a[e][] = -si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi = problem.σ[e, sc, plus½(idx)]
            coeffs.c[e][sc][] = -0.5 * σi
        end
    end
end

function implicit_midpoint_coeffs_nonadjoint_mat!(coeffs, problem, idx, Δϵ)
    ns = n_sums(problem)
    #update the matrix
    for e in 1:ns.ne
        si, sip1 = problem.s[e, minus1(idx)], problem.s[e, idx]
        τi, τip1 = problem.τ[e, minus1(idx)], problem.τ[e, idx]
        coeffs.a[e][] = si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi, σip1 = problem.σ[e, sc, minus1(idx)],  problem.σ[e, sc, idx]
            coeffs.c[e][sc][] = -0.5 * σi
        end
    end
end

function implicit_midpoint_coeffs_adjoint_mat!(coeffs, problem, idx, Δϵ)
    ns = n_sums(problem)
    #update the matrix
    for e in 1:ns.ne
        si = problem.s[e, plus½(idx)]
        τi = problem.τ[e, plus½(idx)]
        coeffs.a[e][] = si / Δϵ + 0.5 * τi
        for sc in 1:ns.nσ
            σi = problem.σ[e, sc, plus½(idx)]
            coeffs.c[e][sc][] = -0.5 * σi
        end
    end
end

function step_nonadjoint!(x, system::DiscretePNSystem2, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs (we multiply the whole linear system with Δϵ -> "normalization")
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, true)
    mul!(system.rhs, system.BM, x, -1.0, true)

    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM⁻¹)
    mul!(x, system.BM⁻¹, system.rhs)
end

function step_adjoint!(x, system::DiscretePNSystem2, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if !system.adjoint @warn "Trying to step_adjoint with system marked as nonadjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(pnsystem.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs
    implicit_midpoint_coeffs_adjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, plus½(idx), -Δϵ, true)
    mul!(system.rhs, transpose(system.BM), x, -1.0, true)

    implicit_midpoint_coeffs_adjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM⁻¹)
    mul!(x, transpose(system.BM⁻¹), system.rhs)
end

function allocate_solution_vector(system::DiscretePNSystem2)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end
