function pnschur() end

Base.@kwdef @concrete struct DiscretePNSchurSystem <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs

    BM
    BM⁻¹
    
    rhs
end

function implicit_midpoint2(pbl::DiscretePNProblem, ::typeof(pnschur), inner_solver::Union{typeof(Krylov.minres), typeof(Krylov.gmres), typeof(\)})
    arch = architecture(pbl)

    coeffs, A, B, C, D = build_coeffs_and_mat_blocks(pbl)
    

    lazy_block_matrix = [A B
    C D]

    lazy_BM⁻¹ = EPMAfem.lazy((PNLazyMatrices.schur_complement, Krylov.minres), lazy_block_matrix)

    BM, BM⁻¹ = unlazy((lazy_block_matrix, lazy_BM⁻¹), vec_size -> allocate_vec(arch, vec_size))

    rhs = allocate_vec(arch, size(lazy_block_matrix, 1))

    return DiscretePNSchurSystem(
        problem = pbl,
        coeffs = coeffs,
        BM = BM,
        BM⁻¹ = BM⁻¹,
        rhs = rhs
    )
end

function step_nonadjoint!(x, system::DiscretePNSchurSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs (we multiply the whole linear system with Δϵ -> "normalization")
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, true)
    mul!(system.rhs, system.BM, x, -1.0, true)

    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)

    mul!(x, system.BM⁻¹, system.rhs, true, false)
end

function step_adjoint!(x, system::DiscretePNSchurSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if !system.adjoint @warn "Trying to step_adjoint with system marked as nonadjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(pnsystem.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs
    implicit_midpoint_coeffs_adjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, plus½(idx), -Δϵ, true)
    mul!(system.rhs, transpose(system.BM), x, -1.0, true)

    implicit_midpoint_coeffs_adjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.BM)

    mul!(y, transpose(system.BM⁻¹), x, true, false)


    linsolve!(system.lin_solver, x, transpose(system.ABCD), system.rhs)
end

function allocate_solution_vector(system::DiscretePNSchurSystem)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    allocate_vec(arch, nxp*nΩp + nxm*nΩm)
end
