abstract type PNImplicitMidpointSolver{T} <: PNSolver{T} end

function energy_step(pn_solv::PNImplicitMidpointSolver)
    pn_equ = get_pn_equ(pn_solv)
    Iϵ = energy_interval(pn_equ)
    return (Iϵ[2] - Iϵ[1])/(pn_solv.N-1)
end

function update_coefficients_rhs_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ)
    pn_equ = get_pn_equ(pn_solv)
    for e in 1:number_of_elements(pn_equ)
        pn_solv.a[e] = -s(pn_equ, ϵip1, e) - s(pn_equ, ϵ2, e) + 0.5 * Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_solv.c[e][i] = -0.5 * Δϵ * σ(pn_equ, ϵ2, e, i)
        end
    end
    b = 0.5*Δϵ
    return pn_solv.a, b, pn_solv.c
end

function update_coefficients_rhs_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ)
    pn_equ = get_pn_equ(pn_solv)
    for e in 1:number_of_elements(pn_equ)
        pn_solv.a[e] = - s(pn_equ, ϵi, e) - s(pn_equ, ϵ2, e) + 0.5 * Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_solv.c[e][i] = -0.5 * Δϵ * σ(pn_equ, ϵ2, e, i)
        end
    end
    b = 0.5*Δϵ
    return pn_solv.a, b, pn_solv.c
end

function update_coefficients_mat_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ)
    pn_equ = get_pn_equ(pn_solv)
    for e in 1:number_of_elements(pn_equ)
        pn_solv.a[e] = s(pn_equ, ϵi, e) + s(pn_equ, ϵ2, e) + 0.5*Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_solv.c[e][i] = -0.5*Δϵ*σ(pn_equ, ϵ2, e, i)
        end
    end
    b = 0.5*Δϵ 
    return pn_solv.a, b, pn_solv.c
end

function update_coefficients_mat_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ)
    pn_equ = get_pn_equ(pn_solv)
    for e in 1:number_of_elements(pn_equ)
        pn_solv.a[e] = s(pn_equ, ϵip1, e) + s(pn_equ, ϵ2, e) + 0.5*Δϵ * τ(pn_equ, ϵ2, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_solv.c[e][i] = -0.5*Δϵ*σ(pn_equ, ϵ2, e, i)
        end
    end
    b = 0.5*Δϵ 
    return pn_solv.a, b, pn_solv.c
end

function update_rhs_forward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ, g_idx)
    pn_semi = pn_solv.pn_semi

    # minus because we have to bring b to the right side of the equation 
    assemble_beam_rhs!(pn_solv.rhs, pn_semi, ϵ2, g_idx, -Δϵ)
    a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
    mul!(pn_solv.rhs, A, current_solution(pn_solv), -1.0, 1.0)
    return
end

function update_rhs_backward!(pn_solv::PNImplicitMidpointSolver, ϵi, ϵ2, ϵip1, Δϵ, μ_idx)
    pn_semi = pn_solv.pn_semi

    # minus because we have to bring b to the right side of the equation 
    assemble_extraction_rhs!(pn_solv.rhs, pn_semi, ϵ2, μ_idx, -Δϵ)
    a, b, c = update_coefficients_rhs_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
    mul!(pn_solv.rhs, A, current_solution(pn_solv), -1.0, 1.0)
    return
end

struct PNFullImplicitMidpointSolver{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}, Tsolv} <: PNImplicitMidpointSolver{T}
    pn_semi::Tpnsemi
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    rhs::V
    lin_solver::Tsolv
    N::Int64
end

function get_pn_equ(pn_solv::PNFullImplicitMidpointSolver)
    return pn_solv.pn_semi.pn_equ
end

function initialize!(pn_solv::PNFullImplicitMidpointSolver{T}) where T
    fill!(pn_solv.lin_solver.x, zero(T))
end

function current_solution(solv::PNFullImplicitMidpointSolver)
    return solv.lin_solver.x
end

# function solve!(pn_solv::PNFullImplicitMidpointSolver{T}) where T
function step_forward!(pn_solv::PNFullImplicitMidpointSolver{T}, ϵi, ϵip1, g_idx) where T
    pn_semi = pn_solv.pn_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    update_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, g_idx)

    a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
    Krylov.solve!(pn_solv.lin_solver, A, pn_solv.rhs, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
end

function step_backward!(pn_solv::PNFullImplicitMidpointSolver{T}, ϵi, ϵip1, μ_idx) where T
    pn_semi = pn_solv.pn_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    update_rhs_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, μ_idx)

    a, b, c = update_coefficients_mat_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
    Krylov.solve!(pn_solv.lin_solver, A, pn_solv.rhs, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
end

function pn_fullimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    return PNFullImplicitMidpointSolver(
        pn_semi,
        ones(T, number_of_elements(equations(pn_semi))),
        [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))], 
        V(undef, max(nLp, nLm)*max(nRp, nRm)),
        V(undef, max(nRp, nRm)),
        V(undef, n),
        Krylov.MinresSolver(n, n, V),
        N,
    )
end

struct PNSchurImplicitMidpointSolver{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}, Tsolv} <: PNImplicitMidpointSolver{T}
    pn_semi::Tpnsemi
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    tmp3::V
    D::V

    rhs_schur::V
    rhs::V
    sol::V
    lin_solver::Tsolv
    N::Int64
end

function pn_schurimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    n = np + nLm*nRm
    return PNSchurImplicitMidpointSolver(
        pn_semi,
        ones(T, number_of_elements(equations(pn_semi))),
        [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))],
        V(undef, max(nLp, nLm)*max(nRp, nRm)),
        V(undef, max(nRp, nRm)),
        V(undef, nLm*nRm),
        V(undef, nLm*nRm),
        V(undef, np),
        V(undef, n),
        V(undef, n),
        Krylov.MinresSolver(np, np, V),
        N
    )
end

function get_pn_equ(pn_solv::PNSchurImplicitMidpointSolver)
    return pn_solv.pn_semi.pn_equ
end

function initialize!(pn_solv::PNSchurImplicitMidpointSolver{T}) where T
    fill!(pn_solv.sol, zero(T))
end

function current_solution(pn_solv::PNSchurImplicitMidpointSolver)
    return pn_solv.sol
end

function step_forward!(pn_solv::PNSchurImplicitMidpointSolver{T}, ϵi, ϵip1, g_idx) where T
    pn_semi = pn_solv.pn_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    update_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, g_idx)

    a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    _update_D(pn_solv, a, b, c)
    _compute_schur_rhs(pn_solv, a, b, c)
    A_schur = SchurBlockMat(pn_semi.ρp, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.kp, pn_semi.Ωpm, pn_semi.absΩp, Diagonal(pn_solv.D), a, b, c, pn_solv.tmp, pn_solv.tmp2, pn_solv.tmp3)
    Krylov.solve!(pn_solv.lin_solver, A_schur, pn_solv.rhs_schur, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
    _compute_full_solution_schur(pn_solv, a, b, c)
end

function step_backward!(pn_solv::PNSchurImplicitMidpointSolver{T}, ϵi, ϵip1, μ_idx) where T
    pn_semi = pn_solv.pn_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    update_rhs_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, μ_idx)

    a, b, c = update_coefficients_mat_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
    _update_D(pn_solv, a, b, c)
    _compute_schur_rhs(pn_solv, a, b, c)
    A_schur = SchurBlockMat(pn_semi.ρp, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.kp, pn_semi.Ωpm, pn_semi.absΩp, Diagonal(pn_solv.D), a, b, c, pn_solv.tmp, pn_solv.tmp2, pn_solv.tmp3)
    Krylov.solve!(pn_solv.lin_solver, A_schur, pn_solv.rhs_schur, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
    _compute_full_solution_schur(pn_solv, a, b, c)
    # @show pn_solv.lin_solver.stats
end

function _update_D(pn_solv::PNSchurImplicitMidpointSolver{T}, a, b, c) where T
    # assemble D
    pn_semi = pn_solv.pn_semi

    ((_, nLm), (_, nRm)) = pn_semi.size
    # tmp_m = @view(pn_solv.tmp[1:nLm*nRm])
    tmp2_m = @view(pn_solv.tmp2[1:nRm])


    fill!(pn_solv.D, zero(T))
    for (ρmz, kmz, az, cz) in zip(pn_semi.ρm, pn_semi.km, a, c)
        tmp2_m .= az*pn_semi.Im.diag
        for (kmzi, czi) in zip(kmz, cz)
            axpy!(czi, kmzi.diag, tmp2_m)
        end

        mul!(reshape(pn_solv.D, (nLm, nRm)), reshape(@view(ρmz.diag[:]), (nLm, 1)), reshape(@view(tmp2_m[:]), (1, nRm)), true, true)
        # axpy!(1.0, tmp_m, pn_solv.D)
    end
end

function _compute_schur_rhs(pn_solv::PNSchurImplicitMidpointSolver, a, b, c)
    pn_semi = pn_solv.pn_semi

    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm

    rhsp = reshape(@view(pn_solv.rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(pn_solv.rhs[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(pn_solv.rhs_schur[:]), (nLp, nRp))

    # A_tmp_m = reshape(@view(pn_solv.tmp3[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(pn_solv.tmp3[1:nLm*nRm]) .= @view(pn_solv.rhs[np+1:np+nm]) ./ pn_solv.D
    # _mul_mp!(rhs_schurp, pn_solv.A_schur.A, A_tmp_m, -1.0)

    mul!(pn_solv.rhs_schur, DMatrix((transpose(∇pmd) for ∇pmd in pn_semi.∇pm), pn_semi.Ωpm, b, mat_view(pn_solv.tmp, nLp, nRm)), @view(pn_solv.tmp3[1:nLm*nRm]), -1.0, true)

end

function _compute_full_solution_schur(pn_solv::PNSchurImplicitMidpointSolver, a, b, c)
    pn_semi = pn_solv.pn_semi

    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm

    full_p = @view(pn_solv.sol[1:np])
    full_m = @view(pn_solv.sol[np+1:np+nm])
    # full_mm = reshape(full_m, (nLm, nRm))

    # bp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    # bm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    full_p .= pn_solv.lin_solver.x

    full_m .= @view(pn_solv.rhs[np+1:np+nm])

    # _mul_pm!(full_mm, pn_solv.A_schur.A, reshape(@view(pn_solv.lin_solver.x[:]), (nLp, nRp)), -1.0)
    mul!(full_m, DMatrix(pn_semi.∇pm, (transpose(Ωpmd) for Ωpmd in pn_semi.Ωpm), b, mat_view(pn_solv.tmp, nLm, nRp)), pn_solv.lin_solver.x, -1.0, true)

    full_m .= full_m ./ pn_solv.D
end