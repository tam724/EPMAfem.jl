abstract type PNImplicitMidpointSolver{T} <: PNSolver{T} end

function update_coefficients_rhs_hightolow!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i
    # i = i-1
    for e in 1:length(solver.a)
        si, sip1 = problem.s[e, i-1], problem.s[e, i]
        s2 = 0.5*(sip1 + si)
        τ2 = 0.5*(problem.τ[e, i-1] + problem.τ[e, i])
        solver.a[e] = -sip1 - s2 + 0.5 * Δϵ * τ2
        for sc in 1:length(solver.c[e])
            σ2 = 0.5*(problem.σ[e, sc, i-1] + problem.σ[e, sc, i])
            solver.c[e][sc] = -0.5 * Δϵ * σ2
        end
    end
    b = 0.5*Δϵ
    return solver.a, b, solver.c
end

function update_coefficients_rhs_lowtohigh!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i+1
    # i = i
    for e in 1:length(solver.a)
        si, sip1 = problem.s[e, i], problem.s[e, i+1]
        s2 = 0.5*(sip1 + si)
        τ2 = 0.5*(problem.τ[e, i] + problem.τ[e, i+1])
        solver.a[e] = - si - s2 + 0.5 * Δϵ * τ2
        for sc in 1:length(solver.c[e])
            σ2 = 0.5*(problem.σ[e, sc, i] + problem.σ[e, sc, i+1])
            solver.c[e][sc] = -0.5 * Δϵ * σ2
        end
    end
    b = 0.5*Δϵ
    return solver.a, b, solver.c
end

function update_coefficients_mat_hightolow!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i
    # i = i-1
    for e in 1:length(solver.a)
        si, sip1 = problem.s[e, i-1], problem.s[e, i]
        s2 = 0.5*(sip1 + si)
        τ2 = 0.5*(problem.τ[e, i-1] + problem.τ[e, i])
        solver.a[e] = si + s2 + 0.5*Δϵ * τ2
        for sc in 1:length(solver.c[e])
            σ2 = 0.5*(problem.σ[e, sc, i-1] + problem.σ[e, sc, i])
            solver.c[e][sc] = -0.5*Δϵ*σ2
        end
    end
    b = 0.5*Δϵ 
    return solver.a, b, solver.c
end

function update_coefficients_mat_lowtohigh!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i+1
    # i = i
    for e in 1:length(solver.a)
        si, sip1 = problem.s[e, i], problem.s[e, i+1]
        s2 = 0.5*(sip1 + si)
        τ2 = 0.5*(problem.τ[e, i] + problem.τ[e, i+1])
        solver.a[e] = sip1 + s2 + 0.5*Δϵ * τ2
        for sc in 1:length(solver.c[e])
            σ2 = 0.5*(problem.σ[e, sc, i] + problem.σ[e, sc, i+1])
            solver.c[e][sc] = -0.5*Δϵ*σ2
        end
    end
    b = 0.5*Δϵ 
    return solver.a, b, solver.c
end

function update_rhs_hightolow!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation 
    gϵ2 = 0.5*(problem.gϵ[i] + problem.gϵ[i-1])
    assemble_rhs_p!(solver.rhs, problem.gxp, problem.gΩp, -Δϵ*gϵ2)
    a, b, c = update_coefficients_rhs_hightolow!(solver, problem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    mul!(solver.rhs, A, current_solution(solver), -1.0, 1.0)
    return
end

function update_rhs_lowtohigh!(solver::PNImplicitMidpointSolver, problem::DiscretePNProblem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation 
    μϵ2 = 0.5*(problem.μϵ[i] + problem.μϵ[i+1])
    assemble_rhs_p!(solver.rhs, problem.μxp, problem.μΩp, -Δϵ*μϵ2)
    a, b, c = update_coefficients_rhs_lowtohigh!(solver, problem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    mul!(solver.rhs, A, current_solution(solver), -1.0, 1.0)
    return
end

struct PNFullImplicitMidpointSolver{T, V<:AbstractVector{T}, Tsolv} <: PNImplicitMidpointSolver{T}
    a::Vector{T}
    c::Vector{Vector{T}}
    tmp::V
    tmp2::V
    rhs::V
    lin_solver::Tsolv
end

function initialize!(pn_solv::PNFullImplicitMidpointSolver{T}) where T
    fill!(pn_solv.lin_solver.x, zero(T))
end

function current_solution(solv::PNFullImplicitMidpointSolver)
    return solv.lin_solver.x
end

function step_hightolow!(solver::PNFullImplicitMidpointSolver{T}, problem::DiscretePNProblem, i, Δϵ) where T
    update_rhs_hightolow!(solver, problem, i, Δϵ)
    a, b, c = update_coefficients_mat_hightolow!(solver, problem, i, Δϵ)
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
end

function step_lowtohigh!(solver::PNFullImplicitMidpointSolver{T}, problem::DiscretePNProblem, i, Δϵ) where T
    update_rhs_lowtohigh!(solver, problem, i, Δϵ)
    a, b, c = update_coefficients_mat_lowtohigh!(solver, problem, i, Δϵ)
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
    Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
end

function pn_fullimplicitmidpointsolver(pn_eq::PNEquations, model::PNGridapModel)
    n = number_of_basis_functions(model)
    n_tot = n.x.p*n.Ω.p + n.x.m*n.Ω.m
    return PNFullImplicitMidpointSolver(
        ones(number_of_elements(pn_eq)),
        [ones(number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)], 
        zeros(max(n.x.p, n.x.m)*max(n.Ω.p, n.Ω.m)),
        zeros(max(n.Ω.p, n.Ω.m)),
        zeros(n_tot),
        Krylov.MinresSolver(n_tot, n_tot, Vector{Float64})
    )
end

function cuda(solver::PNFullImplicitMidpointSolver, T=Float32)
    return PNFullImplicitMidpointSolver(
        Vector{T}(solver.a),
        Vector{T}.(solver.c),
        solver.tmp |> cu,
        solver.tmp2 |> cu,
        solver.rhs |> cu,
        Krylov.MinresSolver(solver.lin_solver.m, solver.lin_solver.m, CuVector{Float32})
    )
end

struct PNSchurImplicitMidpointSolver{T, V<:AbstractVector{T}, Tsolv} <: PNImplicitMidpointSolver{T}
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
end

function pn_schurimplicitmidpointsolver(pn_eq::PNEquations, model::PNGridapModel)
    n = number_of_basis_functions(model)

    np = n.x.p*n.Ω.p
    n_tot = n.x.p*n.Ω.p + n.x.m*n.Ω.m
    return PNSchurImplicitMidpointSolver(
        ones(number_of_elements(pn_eq)),
        [ones(number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)],
        zeros(max(n.x.p, n.x.m)*max(n.Ω.p, n.Ω.m)),
        zeros(max(n.Ω.p, n.Ω.m)),
        zeros(n.x.m*n.Ω.m),
        zeros(n.x.m*n.Ω.m),
        zeros(np),
        zeros(n_tot),
        zeros(n_tot),
        Krylov.MinresSolver(np, np, Vector{Float64}),
    )
end

function cuda(solver::PNSchurImplicitMidpointSolver, T=Float32)
    return PNSchurImplicitMidpointSolver(
        Vector{T}(solver.a),
        Vector{T}.(solver.c),
        solver.tmp |> cu,
        solver.tmp2 |> cu,
        solver.tmp3 |> cu,
        solver.D |> cu,
        solver.rhs_schur |> cu,
        solver.rhs |> cu,
        solver.sol |> cu,
        Krylov.MinresSolver(solver.lin_solver.m, solver.lin_solver.n, CuVector{T})
    )
end


function initialize!(solver::PNSchurImplicitMidpointSolver{T}) where T
    fill!(solver.sol, zero(T))
end

function current_solution(solver::PNSchurImplicitMidpointSolver)
    return solver.sol
end

function step_hightolow!(solver::PNSchurImplicitMidpointSolver{T}, problem::DiscretePNProblem, i, Δϵ) where T
    update_rhs_hightolow!(solver, problem, i, Δϵ)
    a, b, c = update_coefficients_mat_hightolow!(solver, problem, i, Δϵ)
    _update_D(solver, problem, a, b, c)
    _compute_schur_rhs(solver, problem, a, b, c)
    A_schur = SchurBlockMat(problem.ρp, problem.∇pm, problem.∂p, problem.Ip, problem.kp, problem.Ωpm, problem.absΩp, Diagonal(solver.D), a, b, c, solver.tmp, solver.tmp2, solver.tmp3)
    Krylov.solve!(solver.lin_solver, A_schur, solver.rhs_schur, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
    _compute_full_solution_schur(solver, problem, a, b, c)
    return
end

function step_lowtohigh!(solver::PNSchurImplicitMidpointSolver{T}, problem::DiscretePNProblem, i, Δϵ) where T
    update_rhs_lowtohigh!(solver, problem, i, Δϵ)
    a, b, c = update_coefficients_mat_lowtohigh!(solver, problem, i, Δϵ)
    _update_D(solver, problem, a, b, c)
    _compute_schur_rhs(solver, problem, a, b, c)
    A_schur = SchurBlockMat(problem.ρp, problem.∇pm, problem.∂p, problem.Ip, problem.kp, problem.Ωpm, problem.absΩp, Diagonal(solver.D), a, b, c, solver.tmp, solver.tmp2, solver.tmp3)
    Krylov.solve!(solver.lin_solver, A_schur, solver.rhs_schur, rtol=T(1e-14), atol=T(1e-14))
    # @show pn_solv.lin_solver.stats
    _compute_full_solution_schur(solver, problem, a, b, c)
    # @show pn_solv.lin_solver.stats
    return
end

function _update_D(solver::PNSchurImplicitMidpointSolver{T}, problem::DiscretePNProblem, a, b, c) where T
    # assemble D

    (_, (_, nLm), (_, nRm)) = problem.model.n_basis
    # tmp_m = @view(pn_solv.tmp[1:nLm*nRm])
    tmp2_m = @view(solver.tmp2[1:nRm])

    fill!(solver.D, zero(T))
    for (ρmz, kmz, az, cz) in zip(problem.ρm, problem.km, a, c)
        tmp2_m .= az*problem.Im.diag
        for (kmzi, czi) in zip(kmz, cz)
            axpy!(czi, kmzi.diag, tmp2_m)
        end

        mul!(reshape(solver.D, (nLm, nRm)), reshape(@view(ρmz.diag[:]), (nLm, 1)), reshape(@view(tmp2_m[:]), (1, nRm)), true, true)
        # axpy!(1.0, tmp_m, pn_solv.D)
    end
end

function _compute_schur_rhs(solver::PNSchurImplicitMidpointSolver, problem::DiscretePNProblem, a, b, c)

    (_, (nLp, nLm), (nRp, nRm)) = problem.model.n_basis
    
    np = nLp*nRp
    nm = nLm*nRm

    rhsp = reshape(@view(solver.rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(solver.rhs[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(solver.rhs_schur[:]), (nLp, nRp))

    # A_tmp_m = reshape(@view(pn_solv.tmp3[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(solver.tmp3[1:nLm*nRm]) .= @view(solver.rhs[np+1:np+nm]) ./ solver.D
    # _mul_mp!(rhs_schurp, pn_solv.A_schur.A, A_tmp_m, -1.0)

    mul!(solver.rhs_schur, DMatrix((transpose(∇pmd) for ∇pmd in problem.∇pm), problem.Ωpm, b, mat_view(solver.tmp, nLp, nRm)), @view(solver.tmp3[1:nLm*nRm]), -1.0, true)

end

function _compute_full_solution_schur(solver::PNSchurImplicitMidpointSolver, problem::DiscretePNProblem, a, b, c)

    (_, (nLp, nLm), (nRp, nRm)) = problem.model.n_basis

    np = nLp*nRp
    nm = nLm*nRm

    full_p = @view(solver.sol[1:np])
    full_m = @view(solver.sol[np+1:np+nm])
    # full_mm = reshape(full_m, (nLm, nRm))

    # bp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    # bm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    full_p .= solver.lin_solver.x

    full_m .= @view(solver.rhs[np+1:np+nm])

    # _mul_pm!(full_mm, pn_solv.A_schur.A, reshape(@view(pn_solv.lin_solver.x[:]), (nLp, nRp)), -1.0)
    mul!(full_m, DMatrix(problem.∇pm, (transpose(Ωpmd) for Ωpmd in problem.Ωpm), b, mat_view(solver.tmp, nLm, nRp)), solver.lin_solver.x, -1.0, true)

    full_m .= full_m ./ solver.D
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
        assemble_rhs_p!(rhs_K, pn_semi.gx[gj], gΩV, -Δϵ*_excitation_energy_distribution(pn_equ, gi)(ϵ2))
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
        U_hatp = qr([K.Up U.Up]).Q |> mat_type(pn_solv.pn_semi)
        U_hatm = qr([K.Um U.Um]).Q |> mat_type(pn_solv.pn_semi)

        M_hatp = transpose(U_hatp)*U.Up
        M_hatm = transpose(U_hatm)*U.Um

    #L-step
    U = view_U(pn_solv.sol[1], (nLp, nLm), (rp, rm))
    update_U!(pn_proj_semi, pn_semi, U)
    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, _, _ = U_views(pn_proj_semi)
    lin_solver_L = get_lin_solver(pn_solv.lin_solver, nRp*rp + nRm*rm, nRp*rp + nRm*rm)
    # assemble rhs
        rhs_U = cuview(pn_solv.rhs, 1:nRp*rp+nRm*rm)
        # minus because we have to bring b to the right side of the equation
        gxU = gxU_view(pn_proj_semi, gj)
        assemble_rhs_p!(rhs_U, gxU, pn_semi.gΩ[gk], -Δϵ*_excitation_energy_distribution(pn_equ, gi)(ϵ2))
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
        V_hatp = qr([transpose(Lt.Vtp) transpose(Vt.Vtp)]).Q |> mat_type(pn_solv.pn_semi)
        V_hatm = qr([transpose(Lt.Vtm) transpose(Vt.Vtm)]).Q |> mat_type(pn_solv.pn_semi)
        N_hatTp = Vt.Vtp*V_hatp
        N_hatTm = Vt.Vtm*V_hatm
        
    #S-step
    update_Vt!(pn_proj_semi, pn_semi, (Vtp=transpose(V_hatp), Vtm=transpose(V_hatm)))
    VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, _, _ = V_views(pn_proj_semi)
    update_U!(pn_proj_semi, pn_semi, (Up=U_hatp, Um=U_hatm))
    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, _, _ = U_views(pn_proj_semi)
    lin_solver_S = get_lin_solver(pn_solv.lin_solver, 2*rp*2*rp+2*rm*2*rm, 2*rp*2*rp+2*rm*2*rm)
    # assemble rhs
        rhs_S = cuview(pn_solv.rhs, 1:2*rp*2*rp+2*rm*2*rm)
        # minus because we have to bring b to the right side of the equation
        gΩV = gΩV_view(pn_proj_semi, gk)
        gxU = gxU_view(pn_proj_semi, gj)
        assemble_rhs_p!(rhs_S, gxU, gΩV, -Δϵ*_excitation_energy_distribution(pn_equ, gi)(ϵ2))
        a, b, c = update_coefficients_rhs_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        S0_vec = lin_solver_S.x
        S0 = view_S(S0_vec, (2*rp, 2*rm))
        S0_prev = view_S(pn_solv.sol[2], (rp, rm))
        S0.Sp .= M_hatp*S0_prev.Sp*N_hatTp
        S0.Sm .= M_hatm*S0_prev.Sm*N_hatTm
        # minus because we have to bring b to the right side of the equation
        mul!(rhs_S, A, S0_vec, -1.0, 1.0)
        S0_vec = nothing
    # solve the system 
        a, b, c = update_coefficients_mat_forward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
        A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, pn_solv.tmp, pn_solv.tmp2)
        Krylov.minres!(lin_solver_S, A, rhs_S, rtol=T(1e-14), atol=T(1e-14))
        # @show stats
        S_new = view_S(lin_solver_S.x, (2*rp, 2*rm))

        P_hatp, Σ_hatp, Q_hatp = svd(S_new.Sp)
        P_hatm, Σ_hatm, Q_hatm = svd(S_new.Sm)
        r1p, r1m = compute_new_rank(Σ_hatp, pn_solv.max_rank), compute_new_rank(Σ_hatm, pn_solv.max_rank)

    # update the current solution
    U = view_U(pn_solv.sol[1], (nLp, nLm), (r1p, r1m))
    S = view_S(pn_solv.sol[2], (r1p, r1m))
    Vt = view_Vt(pn_solv.sol[3], (nRp, nRm), (r1p, r1m))
    U.Up .= U_hatp*(P_hatp[:, 1:r1p])
    U.Um .= U_hatm*(P_hatm[:, 1:r1m])
    S.Sp .= Diagonal(Σ_hatp[1:r1p])
    S.Sm .= Diagonal(Σ_hatm[1:r1m])
    Vt.Vtp .= transpose(V_hatp*(Q_hatp[:, 1:r1p]))
    Vt.Vtm .= transpose(V_hatm*(Q_hatm[:, 1:r1m]))
    pn_solv.ranks .= [r1p, r1m]
    @show pn_solv.ranks
    return 
end

function compute_new_rank(Σ, max_rank)
    r1 = 1
    Σ = Vector(Σ)
    while sqrt(sum([σ^2 for σ ∈ Σ[r1+1:end]])) > 1e-2
        r1 += 1
    end
    return min(r1, max_rank)
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