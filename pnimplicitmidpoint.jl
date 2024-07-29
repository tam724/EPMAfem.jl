abstract type PNImplicitMidpointSolver{T} <: PNSolver{T} end

function energy_step(pn_solv::PNImplicitMidpointSolver)
    _, _, _, pn_equ = get_mat_b_semi_equ(pn_solv)
    Iϵ = energy_inter(pn_equ)
    return (Iϵ[2] - Iϵ[1])/(pn_solv.N-1)
end

function step_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1, g_idx)
    update_rhs_forward!(pn_solv, ϵi,  ϵip1, g_idx)
    update_mat_forward!(pn_solv, ϵi, ϵip1)
    solve!(pn_solv)
end

function step_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1, μ_idx)
    update_rhs_backward!(pn_solv, ϵi,  ϵip1, μ_idx)
    update_mat_backward!(pn_solv, ϵi, ϵip1)
    solve!(pn_solv)
end

function update_rhs_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1, g_idx)
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    # @assert Δϵ >
    # minus because we have to bring b to the right side of the equation 
    assemble_beam_rhs!(pn_b, pn_semi, ϵ2, g_idx, -Δϵ)

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = -s(pn_equ, ϵip1, e) - s(pn_equ, ϵ2, e) + 0.5 * Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = -0.5 * Δϵ * σ(pn_equ, ϵ2, e, i)
        end
    end
    pn_mat.β[1] = 0.5*Δϵ
    # minus because we have to bring b to the right side of the equation 
    mul!(pn_b, pn_mat, current_solution(pn_solv), -1.0, 1.0)
    return
end

function update_rhs_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1, μ_idx)
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    # @assert Δϵ >
    # minus because we have to bring b to the right side of the equation 
    assemble_extraction_rhs!(pn_b, pn_semi, ϵ2, μ_idx, -Δϵ)

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = - s(pn_equ, ϵi, e) - s(pn_equ, ϵ2, e) + 0.5 * Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = -0.5 * Δϵ * σ(pn_equ, ϵ2, e, i)
        end
    end
    pn_mat.β[1] = 0.5*Δϵ
    # minus because we have to bring b to the right side of the equation 
    mul!(pn_b, pn_mat, current_solution(pn_solv), -1.0, 1.0)
    return
end

function update_mat_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1)
    pn_mat, _, _, pn_equ = get_mat_b_semi_equ(pn_solv)

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = s(pn_equ, ϵi, e) + s(pn_equ, ϵ2, e) + 0.5*Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = -0.5*Δϵ*σ(pn_equ, ϵ2, e, i)
        end
    end
    pn_mat.β[1] = 0.5*Δϵ 
    return
end

function update_mat_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵip1)
    pn_mat, _, _, pn_equ = get_mat_b_semi_equ(pn_solv)

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = s(pn_equ, ϵip1, e) + s(pn_equ, ϵ2, e) + 0.5*Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = -0.5*Δϵ*σ(pn_equ, ϵ2, e, i)
        end
    end
    pn_mat.β[1] = 0.5*Δϵ 
    return
end

struct PNFullImplicitMidpointSolver{T, V<:AbstractVector{T}, Tmat<:PNExplicitImplicitMatrix{T}, Tsolv} <: PNImplicitMidpointSolver{T}
    A::Tmat
    b::V
    lin_solver::Tsolv
    N::Int64
end

function get_mat_b_semi_equ(pn_solv::PNFullImplicitMidpointSolver)
    return (pn_solv.A, pn_solv.b, pn_solv.A.pn_semi, pn_solv.A.pn_semi.pn_equ)
end

function initialize!(pn_solv::PNFullImplicitMidpointSolver{T}) where T
    fill!(pn_solv.lin_solver.x, zero(T))
end

function current_solution(solv::PNFullImplicitMidpointSolver)
    return solv.lin_solver.x
end

function solve!(pn_solv::PNFullImplicitMidpointSolver{T}) where T
    Krylov.solve!(pn_solv.lin_solver, pn_solv.A, pn_solv.b)
end

function pn_fullimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    return PNFullImplicitMidpointSolver(
        pn_explicitimplicitmatrix(pn_semi),
        V(undef, n),
        Krylov.MinresSolver(n, n, V),
        N,
    )
end

struct PNSchurImplicitMidpointSolver{T, V<:AbstractVector{T}, Tmat<:PNSchurImplicitMatrix{T, V}, Tsolv} <: PNImplicitMidpointSolver{T}
    A_schur::Tmat
    b_schur::V
    b::V
    sol::V
    lin_solver::Tsolv
    N::Int64
end

function pn_schurimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    n = np + nLm*nRm
    A_full = pn_explicitimplicitmatrix(pn_semi)
    A_schur = pn_schurimplicitmatrix(A_full)
    return PNSchurImplicitMidpointSolver(
        A_schur,
        V(undef, np),
        V(undef, n),
        V(undef, n),
        Krylov.MinresSolver(np, np, V),
        N,
    )
end

function get_mat_b_semi_equ(pn_solv::PNSchurImplicitMidpointSolver)
    return (pn_solv.A_schur.A, pn_solv.b, pn_solv.A_schur.A.pn_semi, pn_solv.A_schur.A.pn_semi.pn_equ)
end

function initialize!(pn_solv::PNSchurImplicitMidpointSolver{T}) where T
    fill!(pn_solv.sol, zero(T))
end

function current_solution(pn_solv::PNSchurImplicitMidpointSolver)
    return pn_solv.sol
end

function solve!(pn_solv::PNSchurImplicitMidpointSolver{T}) where T
    _update_D(pn_solv.A_schur)
    _compute_schur_rhs(pn_solv)
    Krylov.solve!(pn_solv.lin_solver, pn_solv.A_schur, pn_solv.b_schur, rtol=T(1e-14), atol=T(1e-14))
    _compute_full_solution_schur(pn_solv)
end

function _compute_schur_rhs(pn_solv::PNSchurImplicitMidpointSolver)
    ((nLp, nLm), (nRp, nRm)) = pn_solv.A_schur.A.pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm

    rhsp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    rhsm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(pn_solv.b_schur[:]), (nLp, nRp))

    A_tmp_m = reshape(@view(pn_solv.A_schur.tmp[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(pn_solv.A_schur.tmp[1:nLm*nRm]) .= @view(pn_solv.b[np+1:np+nm]) ./ pn_solv.A_schur.D
    _mul_mp!(rhs_schurp, pn_solv.A_schur.A, A_tmp_m, -1.0)
end

function _compute_full_solution_schur(pn_solv::PNSchurImplicitMidpointSolver)
    ((nLp, nLm), (nRp, nRm)) = pn_solv.A_schur.A.pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm

    full_p = @view(pn_solv.sol[1:np])
    full_m = @view(pn_solv.sol[np+1:np+nm])
    full_mm = reshape(full_m, (nLm, nRm))

    # bp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    # bm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    full_p .= pn_solv.lin_solver.x

    full_m .= @view(pn_solv.b[np+1:np+nm])
    _mul_pm!(full_mm, pn_solv.A_schur.A, reshape(@view(pn_solv.lin_solver.x[:]), (nLp, nRp)), -1.0)
    full_m .= full_m ./ pn_solv.A_schur.D
end