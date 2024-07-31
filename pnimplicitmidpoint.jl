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



struct PNDLRFullImplicitMidpointSolver{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}, Tppnsemi} <: PNImplicitMidpointSolver{T}
    pn_semi::Tpnsemi
    pn_proj_semi::Tppnsemi
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    rhs::V
    #lin_solver::Tsolv
    sol::Tuple{V, V, V}
    rank::Int64
    N::Int64
end

function get_pn_equ(pn_solv::PNDLRFullImplicitMidpointSolver)
    return pn_solv.pn_semi.pn_equ
end

function initialize!(pn_solv::PNDLRFullImplicitMidpointSolver{T}) where T
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    r = pn_solv.rank

    ψp0 = rand(nLp, nRp)
    ψm0 = rand(nLm, nRm)
    Up, Sp, Vtp = svd(ψp0)
    Um, Sm, Vtm = svd(ψm0)

    copyto!(@view(pn_solv.sol[1][1:nLp*r]), Up[:, 1:r][:])
    copyto!(@view(pn_solv.sol[1][nLp*r+1:nLp*r+nLm*r]), Um[:, 1:r][:])

    copyto!(@view(pn_solv.sol[2][1:r*r]), Diagonal(zeros(r))[:])
    copyto!(@view(pn_solv.sol[2][r*r+1:r*r+r*r]), Diagonal(zeros(r))[:])

    copyto!(@view(pn_solv.sol[3][1:r*nRp]), Vtp[1:r, :][:])
    copyto!(@view(pn_solv.sol[3][r*nRp+1:r*nRp+r*nRm]), Vtm[1:r, :][:])
end

function current_solution(pn_solv::PNDLRFullImplicitMidpointSolver)
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    r = pn_solv.rank
    U = view_U(pn_solv.sol[1], (nLp, nLm), r)
    S = view_S(pn_solv.sol[2], r)
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), r)
    ψp = U.Up * S.Sp * Vt.Vtp
    ψm = U.Um * S.Sm * Vt.Vtm
    return [ψp[:]; ψm[:]]
end

function view_U(u, (nLp, nLm), r)
    return (Up=reshape(@view(u[1:nLp*r]), (nLp, r)), Um=reshape(@view(u[nLp*r+1:nLp*r+nLm*r]), (nLm, r)))
end

function view_S(s, r)
    return (Sp=reshape(@view(s[1:r*r]), (r, r)), Sm=reshape(@view(s[r*r+1:r*r+r*r]), (r, r)))
end

function view_Vt(vt, (nRp, nRm), r)
    return (Vtp=reshape(@view(vt[1:r*nRp]), (r, nRp)), Vtm=reshape(@view(vt[r*nRp+1:r*nRp+r*nRm]), (r, nRm)))
end

function step_forward!(pn_solv::PNDLRFullImplicitMidpointSolver{T, V}, ϵi, ϵip1, (gi, gj, gk)) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    r = pn_solv.rank

    pn_semi = pn_solv.pn_semi
    pn_proj_semi = pn_solv.pn_proj_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    rank = pn_solv.rank
    #K-step
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), r)
    update_Vt!(pn_proj_semi, pn_semi, Vt)
    # assemble rhs
        rhs_K = @view(pn_solv.rhs[1:rank*nLp+rank*nLm])
        # minus because we have to bring b to the right side of the equation
        assemble_rhs!(rhs_K, pn_semi.gx[gj], pn_proj_semi.gΩV[gk], -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_proj_semi.VtIpV, pn_proj_semi.VtImV, pn_proj_semi.VtkpV, pn_proj_semi.VtkmV, pn_proj_semi.VtΩpmV, pn_proj_semi.VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        K0 = V(undef, length(pn_solv.sol[1]))
        fill!(K0, zero(T))
        U = view_U(pn_solv.sol[1], (nLp, nLm), r)
        S = view_S(pn_solv.sol[2], r)
        K = view_U(K0, (nLp, nLm), r)
        mul!(K.Up, U.Up, S.Sp)
        mul!(K.Um, U.Um, S.Sm)
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_K, A, K0, -1.0, 1.0)
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_proj_semi.VtIpV, pn_proj_semi.VtImV, pn_proj_semi.VtkpV, pn_proj_semi.VtkmV, pn_proj_semi.VtΩpmV, pn_proj_semi.VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        K1, stats = Krylov.minres(A, rhs_K, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        K = view_U(K1, (nLp, nLm), r)
        U1p = qr(K.Up).Q |> mat_type(pn_solv.pn_semi)
        U1m = qr(K.Um).Q |> mat_type(pn_solv.pn_semi)
        Mp = transpose(U1p)*U.Up
        Mm = transpose(U1m)*U.Um

    #L-step
    U = view_U(pn_solv.sol[1], (nLp, nLm), r)
    update_U!(pn_proj_semi, pn_semi, U)
    # assemble rhs
        rhs_U = @view(pn_solv.rhs[1:nRp*rank+nRm*rank])
        # minus because we have to bring b to the right side of the equation
        assemble_rhs!(rhs_U, pn_proj_semi.gxU[gj], pn_semi.gΩ[gk], -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_proj_semi.UtρpU, pn_proj_semi.UtρmU, pn_proj_semi.Ut∇pmU, pn_proj_semi.Ut∂pU, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Lt0 = V(undef, length(pn_solv.sol[3]))
        fill!(Lt0, zero(T))
        Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), r)
        S = view_S(pn_solv.sol[2], r)
        Lt = view_Vt(Lt0, (nRp, nRm), r)
        mul!(Lt.Vtp, S.Sp, Vt.Vtp)
        mul!(Lt.Vtm, S.Sm, Vt.Vtm)
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_U, A, Lt0, -1.0, 1.0)
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_proj_semi.UtρpU, pn_proj_semi.UtρmU, pn_proj_semi.Ut∇pmU, pn_proj_semi.Ut∂pU, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Lt1, stats = Krylov.minres(A, rhs_U, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        Lt = view_Vt(Lt1, (nRp, nRm), r)
        V1p = qr(transpose(Lt.Vtp)).Q |> mat_type(pn_solv.pn_semi)
        V1m = qr(transpose(Lt.Vtm)).Q |> mat_type(pn_solv.pn_semi)
        Ntp = Vt.Vtp*V1p
        Ntm = Vt.Vtm*V1m
        
    #S-step
    update_Vt!(pn_proj_semi, pn_semi, (Vtp=transpose(V1p), Vtm=transpose(V1m)))
    update_U!(pn_proj_semi, pn_semi, (Up=U1p, Um=U1m))
    # assemble rhs
        rhs_S = @view(pn_solv.rhs[1:rank*rank+rank*rank])
        # minus because we have to bring b to the right side of the equation
        assemble_rhs!(rhs_S, pn_proj_semi.gxU[gj], pn_proj_semi.gΩV[gk], -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_proj_semi.UtρpU, pn_proj_semi.UtρmU, pn_proj_semi.Ut∇pmU, pn_proj_semi.Ut∂pU, pn_proj_semi.VtIpV, pn_proj_semi.VtImV, pn_proj_semi.VtkpV, pn_proj_semi.VtkmV, pn_proj_semi.VtΩpmV, pn_proj_semi.VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        S0 = V(undef, length(pn_solv.sol[2]))
        fill!(S0, zero(T))
        S0_ = view_S(pn_solv.sol[2], r)
        S = view_S(S0, r)
        S.Sp .= Mp*S0_.Sp*Ntp
        S.Sm .= Mm*S0_.Sm*Ntm
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_S, A, S0, -1.0, 1.0)
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_proj_semi.UtρpU, pn_proj_semi.UtρmU, pn_proj_semi.Ut∇pmU, pn_proj_semi.Ut∂pU, pn_proj_semi.VtIpV, pn_proj_semi.VtImV, pn_proj_semi.VtkpV, pn_proj_semi.VtkmV, pn_proj_semi.VtΩpmV, pn_proj_semi.VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        S1, stats = Krylov.minres(A, rhs_S, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        S_new = view_S(S1, r)

    # update the current solution
    U = view_U(pn_solv.sol[1], (nLp, nLm), r)
    S = view_S(pn_solv.sol[2], r)
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), r)
    U.Up .= U1p
    U.Um .= U1m
    S.Sp .= S_new.Sp
    S.Sm .= S_new.Sm
    Vt.Vtp .= transpose(V1p)
    Vt.Vtm .= transpose(V1m)
    return 
end

# function step_backward!(pn_solv::PNFullImplicitMidpointSolver{T}, ϵi, ϵip1, μ_idx) where T
#     pn_semi = pn_solv.pn_semi

#     ϵ2 = 0.5*(ϵi + ϵip1)
#     Δϵ = ϵip1 - ϵi
#     update_rhs_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, μ_idx)

#     a, b, c = update_coefficients_mat_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
#     A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
#     Krylov.solve!(pn_solv.lin_solver, A, pn_solv.rhs, rtol=T(1e-14), atol=T(1e-14))
#     # @show pn_solv.lin_solver.stats
# end

function pn_dlrfullimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N, rank) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    r2 = 2*rank
    return PNDLRFullImplicitMidpointSolver(
        pn_semi,
        pn_projectedsemidiscretization(pn_semi, rank),
        ones(T, number_of_elements(equations(pn_semi))),
        [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))], 
        V(undef, max(nLp, nLm)*max(nRp, nRm)),
        V(undef, max(nRp*nRp, nRm*nRm)),
        V(undef, n),
        (V(undef, nLp*r2 + nLm*r2), V(undef, r2*r2 + r2*r2), V(undef, r2*nRp + r2*nRm)),
        r2,
        N,
    )
end