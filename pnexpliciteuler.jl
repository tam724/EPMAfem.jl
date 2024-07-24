struct PNExplicitEulerSolver{T, V<:AbstractVector{T}, Tmat<:PNExplicitImplicitMatrix{T, V}, Tsolv} <: PNSolver{T}
    A::Tmat
    b::V
    # sol::V
    lin_solver::Tsolv
    N::Int64
    ϵ_interval::Tuple{Float64, Float64}
    cache::V
end

function step_forward!(pn_solv::PNExplicitEulerSolver{T}, ϵi, ϵip1, g_idx) where T
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm
    Δϵ = ϵip1 - ϵi

    # Bp = reshape(@view(B[1:np]), (nLp, nRp))
    # Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))
    copyto!(pn_solv.cache, current_solution(pn_solv))
    # HALF STEP ODD
    update_rhs_forward!(pn_solv, ϵi+0.5*Δϵ, ϵip1, g_idx)
    update_mat_forward!(pn_solv, ϵi+0.5*Δϵ, ϵip1)
    Krylov.solve!(pn_solv.lin_solver, pn_solv.A, pn_solv.b)
    # recover evens
    copyto!(@view(current_solution(pn_solv)[1:np]), @view(pn_solv.cache[1:np]))
    # cache odds
    copyto!(@view(pn_solv.cache[np+1:np+nm]), @view(current_solution(pn_solv)[np+1:np+nm]))
    # FULL STEP EVEN
    update_rhs_forward!(pn_solv, ϵi, ϵip1, g_idx)
    update_mat_forward!(pn_solv, ϵi, ϵip1)
    Krylov.solve!(pn_solv.lin_solver, pn_solv.A, pn_solv.b)
    # recover odds
    copyto!(@view(current_solution(pn_solv)[np+1:np+nm]), @view(pn_solv.cache[np+1:np+nm]))
    # cache evens
    copyto!(@view(pn_solv.cache[1:np]), @view(current_solution(pn_solv)[1:np]))
    # HALF STEP ODD
    update_rhs_forward!(pn_solv, ϵi, ϵip1-0.5*Δϵ, g_idx)
    update_mat_forward!(pn_solv, ϵi, ϵip1-0.5*Δϵ)
    Krylov.solve!(pn_solv.lin_solver, pn_solv.A, pn_solv.b)
    # recover evens
    copyto!(@view(current_solution(pn_solv)[1:np]), @view(pn_solv.cache[1:np]))
end

function energy_step(pn_solv::PNExplicitEulerSolver)
    return (pn_solv.ϵ_interval[2] - pn_solv.ϵ_interval[1])/(pn_solv.N-1)
end

function update_rhs_forward!(pn_solv::PNExplicitEulerSolver, ϵi, ϵip1, g_idx)
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)

    Δϵ = ϵip1 - ϵi
    assemble_beam_rhs!(pn_b, pn_semi, ϵip1, g_idx, -Δϵ)

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = -s(pn_equ, ϵip1, e) - s(pn_equ, ϵip1, e) + Δϵ * τ(pn_equ, ϵip1, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = -Δϵ * σ(pn_equ, ϵip1, e, i)
        end
    end
    pn_mat.β[1] = Δϵ
    # minus because we have to bring b to the right side of the equation 
    mul!(pn_b, pn_mat, current_solution(pn_solv), -1.0, 1.0)
    return
end

function update_mat_forward!(pn_solv::PNExplicitEulerSolver, ϵi, ϵip1)
    pn_mat, _, _, pn_equ = get_mat_b_semi_equ(pn_solv)

    Δϵ = ϵip1 - ϵi

    for e in 1:number_of_elements(pn_equ)
        pn_mat.α[e] = s(pn_equ, ϵi, e) + s(pn_equ, ϵi, e)
        for i in 1:number_of_scatterings(pn_equ)
            pn_mat.γ[e][i] = 0.0
        end
    end
    pn_mat.β[1] = 0.0
    return
end

function get_mat_b_semi_equ(pn_solv::PNExplicitEulerSolver)
    return (pn_solv.A, pn_solv.b, pn_solv.A.pn_semi, pn_solv.A.pn_semi.pn_equ)
end

function initialize!(pn_solv::PNExplicitEulerSolver{T}) where T
    fill!(pn_solv.lin_solver.x, zero(T))
end

function current_solution(pn_solv::PNExplicitEulerSolver)
    return pn_solv.lin_solver.x
end

function pn_expliciteulersolver(pn_semi::PNSemidiscretization{T, V}, ϵ_interval, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    return PNExplicitEulerSolver(
        pn_explicitimplicitmatrix(pn_semi),
        V(undef, n),
        Krylov.MinresSolver(n, n, V),
        N,
        ϵ_interval,
        V(undef, n)
    )
end
