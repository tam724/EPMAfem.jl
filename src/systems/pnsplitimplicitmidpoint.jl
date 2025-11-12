Base.@kwdef @concrete struct DiscretePNSystem3 <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
end

function Base.adjoint(A::DiscretePNSystem3)
    return DiscretePNSystem3(adjoint=!A.adjoint, problem=A.problem, coeffs=A.coeffs, BM=A.BM, BM⁻¹=A.BM⁻¹, rhs=A.rhs)
end

function split_implicit_midpoint(pbl::DiscretePNProblem, solver)
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    Δϵ = step(energy_model(pbl.model))
    
    cfs = (
        Δ=LazyScalar(T(Δϵ)),
        γ=LazyScalar(T(0.5)),
        δ=LazyScalar(T(-0.5)),
        mat = (a = [LazyScalar(zero(T)) for _ in 1:ns.ne], c = [[LazyScalar(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne]),
        rhs = (a = [LazyScalar(zero(T)) for _ in 1:ns.ne], c = [[LazyScalar(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne])
        )
 
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(pbl)

    Ikp(i, cf) = materialize(cf.a[i]*Ip + sum(cf.c[i][j]*kp[i][j] for j in 1:ns.nσ))
    Ikm(i, cf) = materialize(cf.a[i]*Im + sum(cf.c[i][j]*km[i][j] for j in 1:ns.nσ))

    A(cf) = cfs.Δ*(sum(kron_AXB(ρp[i], Ikp(i, cf)) for i in 1:ns.ne) + sum(kron_AXB(∂p[i],cfs.γ * absΩp[i]) for i in 1:ns.nd))
    C(cf) = -(cfs.Δ*sum(kron_AXB(ρm[i], Ikm(i, cf)) for i in 1:ns.ne))
    B = cfs.Δ*(cfs.δ*sum(kron_AXB(∇pm[i], Ωpm[i]) for i in 1:ns.nd))

    inv_matA = solver(A(cfs.mat))
    inv_matC = solver(C(cfs.mat))

    rhsA = A(cfs.rhs)
    rhsC = C(cfs.rhs)

    mats = (
        inv_matA = inv_matA, inv_matC=inv_matC, rhsA=rhsA, rhsC=rhsC, B=B
    )

    ucfs, umats = unlazy((cfs, mats), vec_size -> allocate_vec(arch, vec_size))

    rhs = allocate_vec(arch, size(BM, 1))

    return DiscretePNSystem3(
        problem = pbl,
        coeffs = ucfs,
        mats = umats,
        rhs=rhs
    )
end

function splitting_coeffs_half1(coeffs, problem, idx, Δϵ)
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

function splitting_coeffs_half1(coeffs, problem, idx, Δϵ)
    ns = n_sums(problem)
    #update the matrix
    for e in 1:ns.ne
        si, sip1 = problem.s[e, minus1(idx)], problem.s[e, idx]
        τi, τip1 = problem.τ[e, minus1(idx)], problem.τ[e, idx]
        coeffs.a[e][] = (sip1 + si) / (2*Δϵ) + 0.5 * (τi + (τi + τip1) / 2)
        for sc in 1:ns.nσ
            σi, σip1 = problem.σ[e, sc, minus1(idx)],  problem.σ[e, sc, idx]
            coeffs.c[e][sc][] = -0.5 * (σi + (σi + σip1)/2)
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

function step_nonadjoint!(x, system::DiscretePNSystem3, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs (we multiply the whole linear system with Δϵ -> "normalization")
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, true)
    mul!(system.rhs, system.BM, x, -1.0, true)

    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    mul!(x, system.BM⁻¹, system.rhs)
end

function allocate_solution_vector(system::DiscretePNSystem3)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end
