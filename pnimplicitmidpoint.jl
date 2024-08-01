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
    return
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
    return
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

struct PNDLRFullImplicitMidpointSolver{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}, Tppnsemi, Tsolv} <: PNImplicitMidpointSolver{T}
    pn_semi::Tpnsemi
    pn_proj_semi::Tppnsemi
    a::Vector{T}
    c::Vector{Vector{T}}
    Mbuf::V
    Nbuf::V
    tmp::V
    tmp2::V
    rhs::V
    lin_solver::Tsolv
    sol::Tuple{V, V, V}
    ranks::Vector{Int64}
    max_rank::Int64
    N::Int64
end

function get_pn_equ(pn_solv::PNDLRFullImplicitMidpointSolver)
    return pn_solv.pn_semi.pn_equ
end

function initialize!(pn_solv::PNDLRFullImplicitMidpointSolver{T}) where T
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    rp, rm = pn_solv.ranks

    ψp0 = rand(nLp, nRp)
    ψm0 = rand(nLm, nRm)
    Up, Sp, Vtp = svd(ψp0)
    Um, Sm, Vtm = svd(ψm0)

    copyto!(@view(pn_solv.sol[1][1:nLp*rp]), Up[:, 1:rp][:])
    copyto!(@view(pn_solv.sol[1][nLp*rp+1:nLp*rp+nLm*rm]), Um[:, 1:rm][:])

    copyto!(@view(pn_solv.sol[2][1:rp*rp]), Diagonal(zeros(rp))[:])
    copyto!(@view(pn_solv.sol[2][rp*rp+1:rp*rp+rm*rm]), Diagonal(zeros(rm))[:])

    copyto!(@view(pn_solv.sol[3][1:rp*nRp]), Vtp[1:rp, :][:])
    copyto!(@view(pn_solv.sol[3][rp*nRp+1:rp*nRp+rm*nRm]), Vtm[1:rm, :][:])
end

function current_solution(pn_solv::PNDLRFullImplicitMidpointSolver)
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    rp, rm = pn_solv.ranks
    U = view_U(pn_solv.sol[1], (nLp, nLm), (rp, rm))
    S = view_S(pn_solv.sol[2], (rp, rm))
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), (rp, rm))
    ψp = U.Up * S.Sp * Vt.Vtp
    ψm = U.Um * S.Sm * Vt.Vtm
    return [ψp[:]; ψm[:]]
end

function view_U(u, (nLp, nLm), (rp, rm))
    return (Up=reshape(@view(u[1:nLp*rp]), (nLp, rp)), Um=reshape(@view(u[nLp*rp+1:nLp*rp+nLm*rm]), (nLm, rm)))
end

function view_S(s, (rp, rm))
    return (Sp=reshape(@view(s[1:rp*rp]), (rp, rp)), Sm=reshape(@view(s[rp*rp+1:rp*rp+rm*rm]), (rm, rm)))
end

function view_Vt(vt, (nRp, nRm), (rp, rm))
    return (Vtp=reshape(@view(vt[1:rp*nRp]), (rp, nRp)), Vtm=reshape(@view(vt[rp*nRp+1:rp*nRp+rm*nRm]), (rm, nRm)))
end

function view_M(m, (rp, rm))
    return (Mp=reshape(@view(m[1:r*r]), (r, r)), Mm=reshape(@view(m[r*r+1:r*r+r*r]), (r, r)))
end

function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:CuArray{T, 1, CUDA.DeviceMemory}}
    ## pull the solver internal caches from the "big solver", that is stored in the type
    fill!(bs.err_vec, zero(T))
    stats = bs.stats
    stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
    return Krylov.MinresSolver{T, FC, S}(m, n, @view(bs.Δx[1:0]), @view(bs.x[1:n]), @view(bs.r1[1:n]), @view(bs.r2[1:n]), @view(bs.w1[1:n]), @view(bs.w2[1:n]), @view(bs.y[1:n]), @view(bs.v[1:0]), bs.err_vec, false, stats)
end

function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:Vector{T}}
    ## pull the solver internal caches from the "big solver", that is stored in the type
    fill!(bs.err_vec, zero(T))
    stats = bs.stats
    stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
    return Krylov.MinresSolver{T, FC, UnsafeArray{T, 1}}(m, n, uview(bs.Δx,1:0), uview(bs.x, 1:n), uview(bs.r1, 1:n), uview(bs.r2, 1:n), uview(bs.w1, 1:n), uview(bs.w2, 1:n), uview(bs.y, 1:n), uview(bs.v, 1:0), bs.err_vec, false, stats)
end

cuview(A::Array, slice) = uview(A, slice)
cuview(A::CuArray, slice) = view(A, slice)

function step_forward!(pn_solv::PNDLRFullImplicitMidpointSolver{T, V}, ϵi, ϵip1, (gi, gj, gk)) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_solv.pn_semi.size
    rp, rm = pn_solv.ranks

    pn_semi = pn_solv.pn_semi
    pn_proj_semi = pn_solv.pn_proj_semi

    ϵ2 = 0.5*(ϵi + ϵip1)
    Δϵ = ϵip1 - ϵi
    #K-step
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), (rp, rm))
    update_Vt!(pn_proj_semi, pn_semi, Vt)
    VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, _, _ = V_views(pn_proj_semi)
    lin_solver_K = get_lin_solver(pn_solv.lin_solver, rp*nLp + rm*nLm, rp*nLp + rm*nLm)
    # assemble rhs
        rhs_K = cuview(pn_solv.rhs,1:rp*nLp+rm*nLm)
        # minus because we have to bring b to the right side of the equation
        gΩV = gΩV_view(pn_proj_semi, gk)
        assemble_rhs!(rhs_K, pn_semi.gx[gj], gΩV, -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        # we use the solution buffer of the solver for K0
        K0_vec = lin_solver_K.x
        U = view_U(pn_solv.sol[1], (nLp, nLm), (rp, rm))
        S = view_S(pn_solv.sol[2], (rp, rm))
        K0 = view_U(K0_vec, (nLp, nLm), (rp, rm))
        mul!(K0.Up, U.Up, S.Sp)
        mul!(K0.Um, U.Um, S.Sm)
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_K, A, K0_vec, -1.0, 1.0)
        K0 = nothing # forget about K0, it will be overwritten by the solve
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Krylov.minres!(lin_solver_K, A, rhs_K, rtol=T(1e-14), atol=T(1e-14))
        # K1, stats = Krylov.minres(A, rhs_K, rtol=T(1e-14), atol=T(1e-14))
        @show lin_solver_K.stats
        K = view_U(lin_solver_K.x, (nLp, nLm), (rp, rm))
        @show size(K.Up), size(U.Up)
        U1p = qr(K.Up).Q |> mat_type(pn_solv.pn_semi)
        U1m = qr(K.Um).Q |> mat_type(pn_solv.pn_semi)

        Mp = transpose(U1p)*U.Up
        @show size(Mp)
        Mm = transpose(U1m)*U.Um

    #L-step
    U = view_U(pn_solv.sol[1], (nLp, nLm), (rp, rm))
    update_U!(pn_proj_semi, pn_semi, U)
    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, _, _ = U_views(pn_proj_semi)
    lin_solver_L = get_lin_solver(pn_solv.lin_solver, nRp*rp + nRm*rm, nRp*rp + nRm*rm)
    # assemble rhs
        rhs_U = cuview(pn_solv.rhs, 1:nRp*rp+nRm*rm)
        # minus because we have to bring b to the right side of the equation
        gxU = gxU_view(pn_proj_semi, gj)
        assemble_rhs!(rhs_U, gxU, pn_semi.gΩ[gk], -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Lt0_vec = lin_solver_L.x
        Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), (rp, rm))
        S = view_S(pn_solv.sol[2], (rp, rm))
        Lt0 = view_Vt(Lt0_vec, (nRp, nRm), (rp, rm))
        mul!(Lt0.Vtp, S.Sp, Vt.Vtp)
        mul!(Lt0.Vtm, S.Sm, Vt.Vtm)
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_U, A, Lt0_vec, -1.0, 1.0)
        Lt0 = nothing
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Krylov.minres!(lin_solver_L, A, rhs_U, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        Lt = view_Vt(lin_solver_L.x, (nRp, nRm), (rp, rm))
        V1p = qr(transpose(Lt.Vtp)).Q |> mat_type(pn_solv.pn_semi)
        V1m = qr(transpose(Lt.Vtm)).Q |> mat_type(pn_solv.pn_semi)
        Ntp = Vt.Vtp*V1p
        Ntm = Vt.Vtm*V1m
        
    #S-step
    update_Vt!(pn_proj_semi, pn_semi, (Vtp=transpose(V1p), Vtm=transpose(V1m)))
    VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, _, _ = V_views(pn_proj_semi)
    update_U!(pn_proj_semi, pn_semi, (Up=U1p, Um=U1m))
    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, _, _ = U_views(pn_proj_semi)
    lin_solver_S = get_lin_solver(pn_solv.lin_solver, rp*rp+rm*rm, rp*rp+rm*rm)
    # assemble rhs
        rhs_S = cuview(pn_solv.rhs, 1:rp*rp+rm*rm)
        # minus because we have to bring b to the right side of the equation
        gΩV = gΩV_view(pn_proj_semi, gk)
        gxU = gxU_view(pn_proj_semi, gj)
        assemble_rhs!(rhs_S, gxU, gΩV, -Δϵ*beam_energy(pn_equ, ϵ2, gi))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        S0_vec = lin_solver_S.x
        S0 = view_S(S0_vec, (rp, rm))
        S0_prev = view_S(pn_solv.sol[2], (rp, rm))
        S0.Sp .= Mp*S0_prev.Sp*Ntp
        S0.Sm .= Mm*S0_prev.Sm*Ntm
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_S, A, S0_vec, -1.0, 1.0)
        S0_vec = nothing
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Krylov.minres!(lin_solver_S, A, rhs_S, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        S_new = view_S(lin_solver_S.x, (rp, rm))

    # update the current solution
    U = view_U(pn_solv.sol[1], (nLp, nLm), (rp, rm))
    S = view_S(pn_solv.sol[2], (rp, rm))
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), (rp, rm))
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

function pn_dlrfullimplicitmidpointsolver(pn_semi::PNSemidiscretization{T, V}, N, max_rank) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    r = 2*max_rank # currently the rank is fixed 2r
    mr2 = 2*max_rank
    return PNDLRFullImplicitMidpointSolver(
        pn_semi,
        pn_projectedsemidiscretization(pn_semi, max_rank),
        ones(T, number_of_elements(equations(pn_semi))),
        [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))],
        V(undef, mr2*r),
        V(undef, mr2*r),
        V(undef, max(nLp, nLm)*max(nRp, nRm)), # this can be smaller
        V(undef, max(nRp*nRp, nRm*nRm)), # this can be smaller
        V(undef, n),
        MinresSolver(max(r*max(nLp+nLm, nRp+nRm), mr2*mr2), max(r*max(nLp+nLm, nRp+nRm), mr2*mr2), V),
        (V(undef, nLp*r + nLm*r), V(undef, r*r + r*r), V(undef, r*nRp + r*nRm)),
        [mr2, mr2],
        max_rank,
        N,
    )
end