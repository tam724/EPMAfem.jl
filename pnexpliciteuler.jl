### OUTDATED .. needs rewrite

struct PNExplicitEulerSolver{T, V<:AbstractVector{T}, Tmat<:PNExplicitImplicitMatrix{T, V}, TLU} <: PNSolver{T}
    A::Tmat
    Dm::V
    Dp::TLU
    b::V
    sol::V
    N::Int64
end

function step_forward!(pn_solv::PNExplicitEulerSolver{T}, ϵi, ϵip1, g_idx) where T
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm
    Δϵ = ϵip1 - ϵi

    # Bp = reshape(@view(B[1:np]), (nLp, nRp))
    # Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

    # HALF STEP ODD (m)
    update_rhs_forward!(pn_solv, ϵi+0.5*Δϵ, ϵip1, g_idx, :m)
    update_mat_forward!(pn_solv, ϵi+0.5*Δϵ, ϵip1)
    _update_Dm(pn_solv)
    ldiv!(reshape(@view(pn_solv.sol[np+1:np+nm]), (nLm, nRm)), Diagonal(pn_solv.Dm), reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm)))

    # FULL STEP EVEN (p)
    update_rhs_forward!(pn_solv, ϵi, ϵip1, g_idx, :p)
    update_mat_forward!(pn_solv, ϵi, ϵip1)
    _update_solve_Dp(pn_solv)
    # _update_Dp(pn_solv)
    # SparseArrays.UMFPACK.umfpack_numeric!(pn_solv.Dp)
    # ldiv!(reshape(@view(pn_solv.sol[1:np]), (nLp, nRp)), pn_solv.Dp, reshape(@view(pn_solv.b[1:np]), (nLp, nRp)))
    
    # HALF STEP ODD (m)
    update_rhs_forward!(pn_solv, ϵi, ϵip1-0.5*Δϵ, g_idx, :m)
    update_mat_forward!(pn_solv, ϵi, ϵip1-0.5*Δϵ)
    _update_Dm(pn_solv)
    ldiv!(reshape(@view(pn_solv.sol[np+1:np+nm]), (nLm, nRm)), Diagonal(pn_solv.Dm), reshape(@view(pn_solv.b[np+1:np+nm]), (nLm, nRm)))
end

function _update_solve_Dp(pn_solv::PNExplicitEulerSolver{T, V}) where {T, V}
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm
    fill!(pn_solv.Dp.nzval, zero(T))
    for (ρpz, αz) in zip(pn_solv.A.pn_semi.ρp, pn_solv.A.α)
        bα = αz != 0
        if bα
            axpy!(αz, ρpz.nzval, pn_solv.Dp.nzval)
        end
    end
    SparseArrays.UMFPACK.umfpack_numeric!(pn_solv.Dp, reuse_numeric=false, q=pn_solv.Dp.q)
    ldiv!(reshape(@view(pn_solv.sol[1:np]), (nLp, nRp)), pn_solv.Dp, reshape(@view(pn_solv.b[1:np]), (nLp, nRp)))
end

function _update_solve_Dp(pn_solv::PNExplicitEulerSolver{T, V}) where {T, V<:AbstractGPUArray}
    # there must be a better way for this ! use ala precompute the lu decomposition and apply it to the whole batch. (i dont know how to do it in cuda right now.)
    pn_mat, pn_b, pn_semi, pn_equ = get_mat_b_semi_equ(pn_solv)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    np = nLp*nRp
    nm = nLm*nRm

    lin_solver, mat = pn_solv.Dp
    Bmat = reshape(@view(pn_solv.b[1:np]), (nLp, nRp))
    Solmat = reshape(@view(pn_solv.sol[1:np]), (nLp, nRp))
    for r in 1:nRp
        copyto!(mat.b, @view(Bmat[:, r]))
        Krylov.solve!(lin_solver, mat, mat.b, rtol=T(1e-14), atol=T(1e-14))
        copyto!(@view(Solmat[:, r]), lin_solver.x)
    end

    # SparseArrays.UMFPACK.umfpack_numeric!(pn_solv.Dp)
    # ldiv!(, pn_solv.Dp, )
end

function _update_Dm(pn_solv::PNExplicitEulerSolver{T}) where T
    fill!(pn_solv.Dm, zero(T))
    for (ρmz, αz) in zip(pn_solv.A.pn_semi.ρm, pn_solv.A.α)
        bα = αz != 0
        if bα
            axpy!(-αz, ρmz.diag, pn_solv.Dm)
        end
    end
end

function energy_step(pn_solv::PNExplicitEulerSolver)
    _, _, _, pn_equ = get_mat_b_semi_equ(pn_solv)
    Iϵ = energy_inter(pn_equ)
    return (Iϵ[2] - Iϵ[1])/(pn_solv.N-1)
end

function update_rhs_forward!(pn_solv::PNExplicitEulerSolver, ϵi, ϵip1, g_idx, onlypm)
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
    mul_only!(pn_b, pn_mat, current_solution(pn_solv), -1.0, 1.0, onlypm)
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
    fill!(pn_solv.sol, zero(T))
end

function current_solution(pn_solv::PNExplicitEulerSolver)
    return pn_solv.sol
end

function pn_expliciteulersolver(pn_semi::PNSemidiscretization{T, V}, N) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    n = nLp*nRp + nLm*nRm
    mat = pn_explicitimplicitmatrix(pn_semi)

    return PNExplicitEulerSolver(
        mat,
        V(undef, nLm),
        construct_UMFPACK_solver(pn_semi, mat),
        V(undef, n),
        V(undef, n),
        N
    )
end

function construct_UMFPACK_solver(pn_semi::PNSemidiscretization{T, V}, _) where {T, V}
    return SparseArrays.UMFPACK.UmfpackLU(pn_semi.ρp[1])
end

function construct_UMFPACK_solver(pn_semi::PNSemidiscretization{T, V}, mat) where {T, V<:AbstractGPUArray}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    return (MinresSolver(nLp, nLp, V), pn_pnexplicitmatrixp(mat))
    # dummy = SparseMatrixCSC(pn_semi.ρp[1])
    # dummy = Float64.(dummy)
    # return (SparseArrays.UMFPACK.UmfpackLU(dummy), [Float64.(Vector(ρpe.nzVal)) for ρpe in pn_semi.ρp])
end

