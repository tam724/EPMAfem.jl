function update_coefficients_rhs_nonadjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i
    # i = i-1
    for e in 1:length(system.a)
        si, sip1 = problem.s[e, i-1], problem.s[e, i]
        τi, τip1 = problem.τ[e, i-1], problem.τ[e, i]
        system.a[e] = -sip1 + 0.5 * Δϵ * τip1
        for sc in 1:length(system.c[e])
            σi, σip1 = problem.σ[e, sc, i-1],  problem.σ[e, sc, i]
            system.c[e][sc] = -0.5 * Δϵ * σip1
        end
    end
    b1 = -0.5*Δϵ
    b2 = 0.5*Δϵ
    d = 0.5*Δϵ
    return system.a, (b1, b2, d), system.c
end

function update_coefficients_rhs_adjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, i, Δϵ)
    # ip12 = i+1
    # im12 = i
    for e in 1:length(system.a)
        si = problem.s[e, i]
        τi = problem.τ[e, i]
        system.a[e] = -si + 0.5 * Δϵ * τi
        for sc in 1:length(system.c[e])
            σi = problem.σ[e, sc, i]
            system.c[e][sc] = -0.5 * Δϵ * σi
        end
    end
    b1 = 0.5*Δϵ
    b2 = -0.5*Δϵ
    d = 0.5*Δϵ
    return system.a, (b1, b2, d), system.c
end

function update_coefficients_mat_nonadjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, i, Δϵ)
    # ip1 = i
    # i = i-1
    for e in 1:length(system.a)
        si, sip1 = problem.s[e, i-1], problem.s[e, i]
        τi, τip1 = problem.τ[e, i-1], problem.τ[e, i]
        system.a[e] = si + 0.5 * Δϵ * τi
        for sc in 1:length(system.c[e])
            σi, σip1 = problem.σ[e, sc, i-1],  problem.σ[e, sc, i]
            system.c[e][sc] = -0.5 * Δϵ * σi
        end
    end
    b1 = -0.5*Δϵ
    b2 = 0.5*Δϵ
    d = 0.5*Δϵ
    return system.a, (b1, b2, d), system.c
end

function update_coefficients_mat_adjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, i, Δϵ)
    # ip12 = i+1
    # im12 = i
    for e in 1:length(system.a)
        si = problem.s[e, i]
        τi = problem.τ[e, i]
        system.a[e] = si + 0.5 * Δϵ * τi
        for sc in 1:length(system.c[e])
            σi = problem.σ[e, sc, i]
            system.c[e][sc] = -0.5 * Δϵ * σi
        end
    end
    b1 = 0.5*Δϵ
    b2 = -0.5*Δϵ
    d = 0.5*Δϵ
    return system.a, (b1, b2, d), system.c
end

function update_rhs_nonadjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, rhs::AbstractDiscretePNVector{false}, i, Δϵ, sym)
    # minus because we have to bring b to the right side of the equation 
    # bϵ2 = 0.5*(rhs.bϵ[i] + rhs.bϵ[i-1])
    assemble_rhs_midpoint!(system.rhs, rhs, i-1, -Δϵ, sym)
    a, b, c = update_coefficients_rhs_nonadjoint!(system, problem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, system.tmp, system.tmp2, sym)
    mul!(system.rhs, A, current_solution(system), -1.0, true)
    return
end

function update_rhs_adjoint!(system::AbstractDiscretePNSystemIM, problem::DiscretePNProblem, rhs::AbstractDiscretePNVector{true}, i, Δϵ, sym)
    # minus because we have to bring b to the right side of the equation 
    # bϵ2 = 0.5*(rhs.bϵ[i] + rhs.bϵ[i+1])
    assemble_rhs!(system.rhs, rhs, i, -Δϵ, sym)
    a, b, c = update_coefficients_rhs_adjoint!(system, problem, i, Δϵ)
    # minus because we have to bring b to the right side of the equation
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, system.tmp, system.tmp2, sym)
    mul!(system.rhs, A, current_solution(system), -1.0, true)
    return
end

# full implicit midpoint solver
@concrete struct DiscretePNSystem_IMF <: AbstractDiscretePNSystemIM
    problem
    a
    c
    tmp
    tmp2
    rhs
    lin_solver
    rtol
    atol
end

function initialize!(pnsystem::DiscretePNSystem_IMF)
    # use initial condition from rhs
    arch = architecture(pnsystem.problem)
    T = base_type(arch)
    fill!(pnsystem.lin_solver.x, zero(T))
end

function initialize_from_state!(pnsystem::DiscretePNSystem_IMF, state)
    copy!(pnsystem.lin_solver.x, state)
end

function current_solution(pnsystem::DiscretePNSystem_IMF)
    return pnsystem.lin_solver.x
end

function step_nonadjoint!(pnsystem::DiscretePNSystem_IMF, rhs::AbstractDiscretePNVector{false}, i, Δϵ)
    problem = pnsystem.problem
    update_rhs_nonadjoint!(pnsystem, problem, rhs, i, Δϵ, true)
    a, b, c = update_coefficients_mat_nonadjoint!(pnsystem, problem, i, Δϵ)
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, pnsystem.tmp, pnsystem.tmp2, true)
    Krylov.solve!(pnsystem.lin_solver, A, pnsystem.rhs, rtol=pnsystem.rtol, atol=pnsystem.atol)
    # @show solver.lin_solver.stats
end

function step_adjoint!(pnsystem::DiscretePNSystem_IMF, rhs::AbstractDiscretePNVector{true}, i, Δϵ)
    problem = pnsystem.problem
    update_rhs_adjoint!(pnsystem, problem, rhs, i, Δϵ, true)
    a, b, c = update_coefficients_mat_adjoint!(pnsystem, problem, i, Δϵ)
    A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, pnsystem.tmp, pnsystem.tmp2, true)
    Krylov.solve!(pnsystem.lin_solver, A, pnsystem.rhs, rtol=pnsystem.rtol, atol=pnsystem.atol)
    # @show solver.lin_solver.stats
end

function fullimplicitmidpointsystem(pnproblem::AbstractDiscretePNProblem, tol=nothing)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(pnproblem)
    (nd, ne, nσ) = n_sums(pnproblem)

    arch = architecture(pnproblem)
    T = base_type(arch)

    if isnothing(tol)
        tol = sqrt(eps(Float64))
    end

    n_tot = nxp*nΩp + nxm*nΩm
    return DiscretePNSystem_IMF(
        pnproblem,
        Vector{T}(undef, ne),
        [Vector{T}(undef, nσ) for _ in 1:ne], 
        allocate_vec(arch, max(nxp, nxm)*max(nΩp, nΩm)),
        allocate_vec(arch, max(nΩp, nΩm)),
        allocate_vec(arch, n_tot),
        Krylov.MinresSolver(n_tot, n_tot, vec_type(arch)),
        T(tol),
        T(0),
    )
end

# struct PNPreconImplicitMidpointSolver{T, V<:AbstractVector{T}, Tsolv} <: PNImplicitMidpointSolver{T}
#     a::Vector{T}
#     c::Vector{Vector{T}}
#     tmp::V
#     tmp2::V
#     rhs::V
#     lin_solver::Tsolv
# end

# function initialize!(pn_solv::PNPreconImplicitMidpointSolver{T}, problem) where T
#     # use initial condition from problem
#     fill!(pn_solv.lin_solver.x, zero(T))
# end

# function current_solution(solv::PNPreconImplicitMidpointSolver)
#     return solv.lin_solver.x
# end

# function step_nonadjoint!(solver::PNPreconImplicitMidpointSolver{T}, problem::DiscretePNProblem, rhs::DiscretePNRHS, i, Δϵ) where T
#     update_rhs_nonadjoint!(solver, problem, rhs, i, Δϵ)
#     a, b, c = update_coefficients_mat_nonadjoint!(solver, problem, i, Δϵ)
#     A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
#     Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
#     @show solver.lin_solver.stats
# end

# function step_adjoint!(solver::PNPreconImplicitMidpointSolver{T}, problem::DiscretePNProblem, rhs::DiscretePNRHS, i, Δϵ) where T
#     update_rhs_adjoint!(solver, problem, rhs, i, Δϵ)
#     a, b, c = update_coefficients_mat_adjoint!(solver, problem, i, Δϵ)
#     A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
#     Krylov.solve!(solver.lin_solver, A, solver.rhs, rtol=T(1e-14), atol=T(1e-14))
#     # @show pn_solv.lin_solver.stats
# end

# function pn_fullimplicitmidpointsolver(pn_eq::PNEquations, discrete_model::PNGridapModel)
#     n = number_of_basis_functions(discrete_model)
#     VT = vec_type(discrete_model)
#     T = base_type(discrete_model)

#     n_tot = n.x.p*n.Ω.p + n.x.m*n.Ω.m
#     return PNPreconImplicitMidpointSolver(
#         Vector{T}(undef, number_of_elements(pn_eq)),
#         [Vector{T}(undef, number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)], 
#         VT(undef, max(n.x.p, n.x.m)*max(n.Ω.p, n.Ω.m)),
#         VT(undef, max(n.Ω.p, n.Ω.m)),
#         VT(undef, n_tot),
#         Krylov.MinresSolver(n_tot, n_tot, VT)
#     )
# end

@concrete struct DiscretePNSystem_IMS <: AbstractDiscretePNSystemIM
    problem
    a
    c
    tmp
    tmp2
    tmp3
    D

    rhs_schur
    rhs
    sol
    lin_solver
    rtol
    atol
end

function schurimplicitmidpointsystem(pnproblem::AbstractDiscretePNProblem, tol=nothing)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(pnproblem)
    (nd, ne, nσ) = n_sums(pnproblem)

    arch = architecture(pnproblem)
    T = base_type(arch)

    if isnothing(tol)
        tol = sqrt(eps(Float64))
    end

    np = nxp*nΩp
    n_tot = nxp*nΩp + nxm*nΩm
    return DiscretePNSystem_IMS(
        pnproblem,
        Vector{T}(undef, ne),
        [Vector{T}(undef, nσ) for _ in 1:ne],
        allocate_vec(arch, max(nxp, nxm)*max(nΩp, nΩm)),
        allocate_vec(arch, max(nΩp, nΩm)),
        allocate_vec(arch, nxm*nΩm),
        allocate_vec(arch, nxm*nΩm),
        allocate_vec(arch, np),
        allocate_vec(arch, n_tot),
        allocate_vec(arch, n_tot),
        Krylov.MinresSolver(np, np, vec_type(arch)),
        T(tol),
        T(0)
    )
end

function initialize!(pnsystem::DiscretePNSystem_IMS)
    # use initial condition from rhs
    arch = architecture(pnsystem.problem)
    T = base_type(arch)
    fill!(pnsystem.sol, zero(T))
end

function initialize_from_state!(pnsystem::DiscretePNSystem_IMS, state)
    copy!(pnsystem.sol, state)
end

function current_solution(pnsystem::DiscretePNSystem_IMS)
    return pnsystem.sol
end

function step_nonadjoint!(pnsystem::DiscretePNSystem_IMS, rhs::AbstractDiscretePNVector{false}, i, Δϵ)
    problem = pnsystem.problem
    update_rhs_nonadjoint!(pnsystem, problem, rhs, i, Δϵ, false)
    a, b, c = update_coefficients_mat_nonadjoint!(pnsystem, problem, i, Δϵ)
    _update_D(pnsystem, a, b, c) # assembles the right lower block (diagonal)
    _compute_schur_rhs(pnsystem, a, b, c) # computes the schur rhs (using the inverse of D)
    A_schur = SchurBlockMat(problem.ρp, problem.∇pm, problem.∂p, problem.Ip, problem.kp, problem.Ωpm, problem.absΩp, Diagonal(pnsystem.D), a, b, c, pnsystem.tmp, pnsystem.tmp2, pnsystem.tmp3)
    Krylov.solve!(pnsystem.lin_solver, A_schur, pnsystem.rhs_schur, rtol=pnsystem.rtol, atol=pnsystem.atol)
    # @show solver.lin_solver.stats
    _compute_full_solution_schur(pnsystem, a, b, c) # reconstructs lower part of the solution vector
    return
end

function step_adjoint!(pnsystem::DiscretePNSystem_IMS, rhs::AbstractDiscretePNVector{true}, i, Δϵ)
    problem = pnsystem.problem
    update_rhs_adjoint!(pnsystem, problem, rhs, i, Δϵ, false)
    a, b, c = update_coefficients_mat_adjoint!(pnsystem, problem, i, Δϵ)
    _update_D(pnsystem, a, b, c)
    _compute_schur_rhs(pnsystem, a, b, c)
    A_schur = SchurBlockMat(problem.ρp, problem.∇pm, problem.∂p, problem.Ip, problem.kp, problem.Ωpm, problem.absΩp, Diagonal(pnsystem.D), a, b, c, pnsystem.tmp, pnsystem.tmp2, pnsystem.tmp3)
    Krylov.solve!(pnsystem.lin_solver, A_schur, pnsystem.rhs_schur, rtol=pnsystem.rtol, atol=pnsystem.atol)
    # @show pn_solv.lin_solver.stats
    _compute_full_solution_schur(pnsystem, a, b, c)
    # @show pn_solv.lin_solver.stats
    return
end

function _update_D(pnsystem::DiscretePNSystem_IMS, a, b, c)
    problem = pnsystem.problem
    # assemble D

    (_, (_, nLm), (_, nRm)) = n_basis(problem.model)
    # tmp_m = @view(pn_solv.tmp[1:nLm*nRm])
    tmp2_m = @view(pnsystem.tmp2[1:nRm])

    fill!(pnsystem.D, zero(eltype(pnsystem.D)))
    for (ρmz, kmz, az, cz) in zip(problem.ρm, problem.km, a, c)
        tmp2_m .= az*problem.Im.diag
        for (kmzi, czi) in zip(kmz, cz)
            axpy!(czi, kmzi.diag, tmp2_m)
        end

        mul!(reshape(pnsystem.D, (nLm, nRm)), reshape(@view(ρmz.diag[:]), (nLm, 1)), reshape(@view(tmp2_m[:]), (1, nRm)), true, true)
        # axpy!(1.0, tmp_m, pn_solv.D)
    end
end

function _compute_schur_rhs(pnsystem::DiscretePNSystem_IMS, a, b, c)
    problem = pnsystem.problem

    (_, (nLp, nLm), (nRp, nRm)) = n_basis(problem.model)
    
    np = nLp*nRp
    nm = nLm*nRm

    (b1, b2, d) = b

    rhsp = reshape(@view(pnsystem.rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(pnsystem.rhs[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(pnsystem.rhs_schur[:]), (nLp, nRp))

    # A_tmp_m = reshape(@view(pn_solv.tmp3[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(pnsystem.tmp3[1:nLm*nRm]) .= @view(pnsystem.rhs[np+1:np+nm]) ./ pnsystem.D
    # _mul_mp!(rhs_schurp, pn_solv.A_schur.A, A_tmp_m, -1.0)

    mul!(pnsystem.rhs_schur, DMatrix(problem.∇pm, problem.Ωpm, b1, mat_view(pnsystem.tmp, nLp, nRm)), @view(pnsystem.tmp3[1:nLm*nRm]), -1.0, true)
end

function _compute_full_solution_schur(pnsystem::DiscretePNSystem_IMS, a, b, c)
    problem = pnsystem.problem

    (_, (nLp, nLm), (nRp, nRm)) = n_basis(problem.model)

    np = nLp*nRp
    nm = nLm*nRm

    (b1, b2, d) = b

    full_p = @view(pnsystem.sol[1:np])
    full_m = @view(pnsystem.sol[np+1:np+nm])
    # full_mm = reshape(full_m, (nLm, nRm))

    # bp = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    # bm = reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm))

    full_p .= pnsystem.lin_solver.x

    full_m .= @view(pnsystem.rhs[np+1:np+nm])

    # _mul_pm!(full_mm, pn_solv.A_schur.A, reshape(@view(pn_solv.lin_solver.x[:]), (nLp, nRp)), -1.0)
    mul!(full_m, DMatrix((transpose(∇pmd) for ∇pmd in problem.∇pm), (transpose(Ωpmd) for Ωpmd in problem.Ωpm), b2, mat_view(pnsystem.tmp, nLm, nRp)), pnsystem.lin_solver.x, -1.0, true)

    full_m .= full_m ./ pnsystem.D
end


### TODO!!!!!
# struct PNDLRFullImplicitMidpointSolver{T, V<:AbstractVector{T}, TPP, Tsolv} <: AbstractDiscretePNSystemIM{T}
#     proj_problem::TPP
#     a::Vector{T}
#     c::Vector{Vector{T}}
#     Mbuf::V
#     Nbuf::V
#     tmp::V
#     tmp2::V
#     rhs::V
#     lin_solver::Tsolv
#     sol::Tuple{V, V, V}
#     ranks::Vector{Int64}
#     max_rank::Int64
# end

# function initialize!(solver::PNDLRFullImplicitMidpointSolver{T}, problem) where T
#     (_, (nLp, nLm), (nRp, nRm)) = n_basis(problem.model)
#     rp, rm = solver.ranks

#     # this is still a bit strange
#     ψp0 = rand(nLp, nRp)
#     ψm0 = rand(nLm, nRm)
#     Up, Sp, Vtp = svd(ψp0)
#     Um, Sm, Vtm = svd(ψm0)

#     copyto!(@view(solver.sol[1][1:nLp*rp]), Up[:, 1:rp][:])
#     copyto!(@view(solver.sol[1][nLp*rp+1:nLp*rp+nLm*rm]), Um[:, 1:rm][:])

#     copyto!(@view(solver.sol[2][1:rp*rp]), Diagonal(zeros(rp))[:])
#     copyto!(@view(solver.sol[2][rp*rp+1:rp*rp+rm*rm]), Diagonal(zeros(rm))[:])

#     copyto!(@view(solver.sol[3][1:rp*nRp]), Vtp[1:rp, :][:])
#     copyto!(@view(solver.sol[3][rp*nRp+1:rp*nRp+rm*nRm]), Vtm[1:rm, :][:])
# end

# # maybe this should not depend on the problem (the solver could have the view_U,S,Vt functions available with the current rank)
# function current_solution(solver::PNDLRFullImplicitMidpointSolver, problem)
#     (_, (nLp, nLm), (nRp, nRm)) = n_basis(problem.model)
#     rp, rm = solver.ranks
#     U = view_U(solver.sol[1], (nLp, nLm), (rp, rm))
#     S = view_S(solver.sol[2], (rp, rm))
#     Vt = view_Vt(solver.sol[3], (nRp, nRm), (rp, rm))
#     ψp = U.Up * S.Sp * Vt.Vtp
#     ψm = U.Um * S.Sm * Vt.Vtm
#     return [ψp[:]; ψm[:]]
# end

# function view_U(u, (nLp, nLm), (rp, rm))
#     return (Up=reshape(@view(u[1:nLp*rp]), (nLp, rp)), Um=reshape(@view(u[nLp*rp+1:nLp*rp+nLm*rm]), (nLm, rm)))
# end

# function view_S(s, (rp, rm))
#     return (Sp=reshape(@view(s[1:rp*rp]), (rp, rp)), Sm=reshape(@view(s[rp*rp+1:rp*rp+rm*rm]), (rm, rm)))
# end

# function view_Vt(vt, (nRp, nRm), (rp, rm))
#     return (Vtp=reshape(@view(vt[1:rp*nRp]), (rp, nRp)), Vtm=reshape(@view(vt[rp*nRp+1:rp*nRp+rm*nRm]), (rm, nRm)))
# end

# function view_M(m, (rp, rm))
#     return (Mp=reshape(@view(m[1:r*r]), (r, r)), Mm=reshape(@view(m[r*r+1:r*r+r*r]), (r, r)))
# end

# function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:CuArray{T, 1}}
#     ## pull the solver internal caches from the "big solver", that is stored in the type
#     fill!(bs.err_vec, zero(T))
#     stats = bs.stats
#     stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
#     return Krylov.MinresSolver{T, FC, S}(m, n, @view(bs.Δx[1:0]), @view(bs.x[1:n]), @view(bs.r1[1:n]), @view(bs.r2[1:n]), @view(bs.w1[1:n]), @view(bs.w2[1:n]), @view(bs.y[1:n]), @view(bs.v[1:0]), bs.err_vec, false, stats)
# end

# function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:Vector{T}}
#     ## pull the solver internal caches from the "big solver", that is stored in the type
#     fill!(bs.err_vec, zero(T))
#     stats = bs.stats
#     stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
#     return Krylov.MinresSolver{T, FC, UnsafeArray{T, 1}}(m, n, uview(bs.Δx,1:0), uview(bs.x, 1:n), uview(bs.r1, 1:n), uview(bs.r2, 1:n), uview(bs.w1, 1:n), uview(bs.w2, 1:n), uview(bs.y, 1:n), uview(bs.v, 1:0), bs.err_vec, false, stats)
# end

# cuview(A::Array, slice) = uview(A, slice)
# cuview(A::CuArray, slice) = view(A, slice)

# function step_nonadjoint!(solver::PNDLRFullImplicitMidpointSolver{T, V}, problem::DiscretePNProblem, rhs::AbstractDiscretePNVector{false}, i, Δϵ) where {T, V}
#     (_, (nLp, nLm), (nRp, nRm)) = n_basis(problem.model)
#     rp, rm = solver.ranks

#     proj_problem = solver.proj_problem

#     # bϵ2 = 0.5*(rhs.bϵ[i] + rhs.bϵ[i-1])

#     #K-step
#     Vt = view_Vt(solver.sol[3], (nRp, nRm), (rp, rm))
#     update_Vt!(proj_problem, problem, rhs, Vt)
#     VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, bΩpV = V_views(proj_problem)
#     lin_solver_K = get_lin_solver(solver.lin_solver, rp*nLp + rm*nLm, rp*nLp + rm*nLm)
#     # assemble rhs
#         rhs_K = cuview(solver.rhs,1:rp*nLp+rm*nLm)
#         # minus because we have to bring b to the right side of the equation
#         # bΩpV = bΩpV_view(proj_problem)
#         assemble_rhs_p_midpoint!(rhs_K, rhs, i-1, -Δϵ; bΩp=bΩpV)
#         a, b, c = update_coefficients_rhs_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, solver.tmp, solver.tmp2)
#         # we use the solution buffer of the solver for K0
#         K0_vec = lin_solver_K.x
#         U = view_U(solver.sol[1], (nLp, nLm), (rp, rm))
#         S = view_S(solver.sol[2], (rp, rm))
#         K0 = view_U(K0_vec, (nLp, nLm), (rp, rm))
#         mul!(K0.Up, U.Up, S.Sp)
#         mul!(K0.Um, U.Um, S.Sm)
#         # minus because we have to bring b to the right side of the equation
#         mul!(rhs_K, A, K0_vec, -1.0, true)
#         K0 = nothing # forget about K0, it will be overwritten by the solve
#     # solve the system 
#         a, b, c = update_coefficients_mat_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(problem.ρp, problem.ρm, problem.∇pm, problem.∂p, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, solver.tmp, solver.tmp2)
#         Krylov.minres!(lin_solver_K, A, rhs_K, rtol=T(1e-14), atol=T(1e-14))
#         # K1, stats = Krylov.minres(A, rhs_K, rtol=T(1e-14), atol=T(1e-14))
#         @show lin_solver_K.stats
#         K = view_U(lin_solver_K.x, (nLp, nLm), (rp, rm))
#         U_hatp = qr([K.Up U.Up]).Q |> mat_type(problem)
#         U_hatm = qr([K.Um U.Um]).Q |> mat_type(problem)

#         M_hatp = transpose(U_hatp)*U.Up
#         M_hatm = transpose(U_hatm)*U.Um

#     #L-step
#     U = view_U(solver.sol[1], (nLp, nLm), (rp, rm))
#     update_U!(proj_problem, problem, rhs, U)
#     UtρpU, UtρmU, Ut∂pU, Ut∇pmU, bxpU = U_views(proj_problem)
#     lin_solver_L = get_lin_solver(solver.lin_solver, nRp*rp + nRm*rm, nRp*rp + nRm*rm)
#     # assemble rhs
#         rhs_U = cuview(solver.rhs, 1:nRp*rp+nRm*rm)
#         # minus because we have to bring b to the right side of the equation
#         # bxpU = bxpU_view(proj_problem)
#         assemble_rhs_p_midpoint!(rhs_U, rhs, i-1, -Δϵ; bxp=bxpU)
#         a, b, c = update_coefficients_rhs_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
#         Lt0_vec = lin_solver_L.x
#         Vt = view_Vt(solver.sol[3], (nRp, nRm), (rp, rm))
#         S = view_S(solver.sol[2], (rp, rm))
#         Lt0 = view_Vt(Lt0_vec, (nRp, nRm), (rp, rm))
#         mul!(Lt0.Vtp, S.Sp, Vt.Vtp)
#         mul!(Lt0.Vtm, S.Sm, Vt.Vtm)
#         # minus because we have to bring b to the right side of the equation
#         mul!(rhs_U, A, Lt0_vec, -1.0, 1.0)
#         Lt0 = nothing
#     # solve the system 
#         a, b, c = update_coefficients_mat_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, problem.Ip, problem.Im, problem.kp, problem.km, problem.Ωpm,  problem.absΩp, a, b, c, solver.tmp, solver.tmp2)
#         Krylov.minres!(lin_solver_L, A, rhs_U, rtol=T(1e-14), atol=T(1e-14))
#         # @show stats
#         Lt = view_Vt(lin_solver_L.x, (nRp, nRm), (rp, rm))
#         V_hatp = qr([transpose(Lt.Vtp) transpose(Vt.Vtp)]).Q |> mat_type(problem)
#         V_hatm = qr([transpose(Lt.Vtm) transpose(Vt.Vtm)]).Q |> mat_type(problem)
#         N_hatTp = Vt.Vtp*V_hatp
#         N_hatTm = Vt.Vtm*V_hatm
        
#     #S-step
#     update_Vt!(proj_problem, problem, rhs, (Vtp=transpose(V_hatp), Vtm=transpose(V_hatm)))
#     VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, bΩpV = V_views(proj_problem)
#     update_U!(proj_problem, problem, rhs, (Up=U_hatp, Um=U_hatm))
#     UtρpU, UtρmU, Ut∂pU, Ut∇pmU, bxpU = U_views(proj_problem)
#     lin_solver_S = get_lin_solver(solver.lin_solver, 2*rp*2*rp+2*rm*2*rm, 2*rp*2*rp+2*rm*2*rm)
#     # assemble rhs
#         rhs_S = cuview(solver.rhs, 1:2*rp*2*rp+2*rm*2*rm)
#         # minus because we have to bring b to the right side of the equation
#         # bΩpV = bΩpV_view(proj_problem)
#         # bxpU = bxpU_view(proj_problem)
#         assemble_rhs_p_midpoint!(rhs_S, rhs, i-1, -Δϵ; bxp=bxpU, bΩp=bΩpV)
#         a, b, c = update_coefficients_rhs_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, solver.tmp, solver.tmp2)
#         S0_vec = lin_solver_S.x
#         S0 = view_S(S0_vec, (2*rp, 2*rm))
#         S0_prev = view_S(solver.sol[2], (rp, rm))
#         S0.Sp .= M_hatp*S0_prev.Sp*N_hatTp
#         S0.Sm .= M_hatm*S0_prev.Sm*N_hatTm
#         # minus because we have to bring b to the right side of the equation
#         mul!(rhs_S, A, S0_vec, -1.0, 1.0)
#         S0_vec = nothing
#     # solve the system 
#         a, b, c = update_coefficients_mat_nonadjoint!(solver, problem, i, Δϵ)
#         A = FullBlockMat(UtρpU, UtρmU, Ut∇pmU, Ut∂pU, VtIpV, VtImV, VtkpV, VtkmV, VtΩpmV, VtabsΩpV, a, b, c, solver.tmp, solver.tmp2)
#         Krylov.minres!(lin_solver_S, A, rhs_S, rtol=T(1e-14), atol=T(1e-14))
#         # @show stats
#         S_new = view_S(lin_solver_S.x, (2*rp, 2*rm))

#         P_hatp, Σ_hatp, Q_hatp = svd(S_new.Sp)
#         P_hatm, Σ_hatm, Q_hatm = svd(S_new.Sm)
#         r1p, r1m = compute_new_rank(Σ_hatp, solver.max_rank), compute_new_rank(Σ_hatm, solver.max_rank)

#     # update the current solution
#     U = view_U(solver.sol[1], (nLp, nLm), (r1p, r1m))
#     S = view_S(solver.sol[2], (r1p, r1m))
#     Vt = view_Vt(solver.sol[3], (nRp, nRm), (r1p, r1m))
#     U.Up .= U_hatp*(P_hatp[:, 1:r1p])
#     U.Um .= U_hatm*(P_hatm[:, 1:r1m])
#     S.Sp .= Diagonal(Σ_hatp[1:r1p])
#     S.Sm .= Diagonal(Σ_hatm[1:r1m])
#     Vt.Vtp .= transpose(V_hatp*(Q_hatp[:, 1:r1p]))
#     Vt.Vtm .= transpose(V_hatm*(Q_hatm[:, 1:r1m]))
#     solver.ranks .= [r1p, r1m]
#     @show solver.ranks
#     return 
# end

# function compute_new_rank(Σ, max_rank)
#     r1 = 1
#     Σ = Vector(Σ)
#     while sqrt(sum([σ^2 for σ ∈ Σ[r1+1:end]])) > 1e-3
#         r1 += 1
#     end
#     return min(r1, max_rank)
# end

# # function step_backward!(pn_solv::PNFullImplicitMidpointSolver{T}, ϵi, ϵip1, μ_idx) where T
# #     pn_semi = pn_solv.pn_semi

# #     ϵ2 = 0.5*(ϵi + ϵip1)
# #     Δϵ = ϵip1 - ϵi
# #     update_rhs_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ, μ_idx)

# #     a, b, c = update_coefficients_mat_backward!(pn_solv, ϵi, ϵ2, ϵip1, Δϵ)
# #     A = FullBlockMat(pn_semi.ρp, pn_semi.ρm, pn_semi.∇pm, pn_semi.∂p, pn_semi.Ip, pn_semi.Im, pn_semi.kp, pn_semi.km, pn_semi.Ωpm,  pn_semi.absΩp, a, b, c, pn_solv.tmp, pn_solv.tmp2)
# #     Krylov.solve!(pn_solv.lin_solver, A, pn_solv.rhs, rtol=T(1e-14), atol=T(1e-14))
# #     # @show pn_solv.lin_solver.stats
# # end

# function pn_dlrfullimplicitmidpointsolver(pn_eq::PNEquations, discrete_model::PNGridapModel, max_rank)
#     (_, (nLp, nLm), (nRp, nRm)) = n_basis(discrete_model)
#     VT = vec_type(discrete_model)
#     T = base_type(discrete_model)

#     n = nLp*nRp + nLm*nRm
#     r = 2*max_rank # currently the rank is fixed 2r
#     mr2 = 2*max_rank
#     proj_problem = pn_projectedproblem(pn_eq, discrete_model, max_rank)

#     return PNDLRFullImplicitMidpointSolver(
#         proj_problem,
#         Vector{T}(undef, number_of_elements(pn_eq)),
#         [Vector{T}(undef, number_of_scatterings(pn_eq)) for _ in 1:number_of_elements(pn_eq)],
#         VT(undef, mr2*r),
#         VT(undef, mr2*r),
#         VT(undef, max(nLp, nLm)*max(nRp, nRm)), # this can be smaller
#         VT(undef, max(nRp*nRp, nRm*nRm)), # this can be smaller
#         VT(undef, n),
#         MinresSolver(max(r*max(nLp+nLm, nRp+nRm), mr2*mr2), max(r*max(nLp+nLm, nRp+nRm), mr2*mr2), VT),
#         (VT(undef, nLp*r + nLm*r), VT(undef, r*r + r*r), VT(undef, r*nRp + r*nRm)),
#         [1, 1],
#         max_rank
#     )
# end