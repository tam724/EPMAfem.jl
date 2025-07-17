Base.@kwdef @concrete struct DiscreteDLRPNSystem <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp

    max_rank
end

function implicit_midpoint_dlr(pbl::DiscretePNProblem; max_rank=20)
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    nb = EPMAfem.n_basis(pbl)
    Δϵ = step(energy_model(pbl.model))

    Vt = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, max_rank, nb.nΩ.p), (Ref(max_rank), Ref(nb.nΩ.p)))
    V = transpose(Vt)
    U = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, nb.nx.p, max_rank), (Ref(nb.nx.p), Ref(max_rank)))
    Ut = transpose(U)

    coeffs = (a = [Ref(zero(T)) for _ in 1:ns.ne], c = [[Ref(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne], Δ=Ref(T(Δϵ)), γ=Ref(T(0.5)), δ=Ref(T(-0.5)), δt=Ref(T(0.5)))
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(pbl)

    A = sum(kron_AXB(ρp[i], coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ)) for i in 1:ns.ne)
    B = sum(kron_AXB(∇pm[i], Ωpm[i]) for i in 1:ns.nd)
    C = sum(kron_AXB(ρm[i], coeffs.a[i]*Im + sum(coeffs.c[i][j]*km[i][j] for j in 1:ns.nσ)) for i in 1:ns.ne)
    D = sum(kron_AXB(∂p[i], absΩp[i]) for i in 1:ns.nd)

    BM = [
        coeffs.Δ*(A + coeffs.γ*D) coeffs.Δ*(coeffs.δ*B)
        T(-1)*(coeffs.Δ*(coeffs.δt*transpose(B))) T(-1)*(coeffs.Δ*C)
    ]

    A_V = sum(kron_AXB(ρp[i], Vt*(coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ))*V) for i in 1:ns.ne)
    B_V = sum(kron_AXB(∇pm[i], Ωpm[i]*V) for i in 1:ns.nd)
    D_V = sum(kron_AXB(∂p[i], Vt*absΩp[i]*V) for i in 1:ns.nd)

    BM_V = [
        coeffs.Δ*(A_V + coeffs.γ*D_V) coeffs.Δ*(coeffs.δ*B_V)
        T(-1)*(coeffs.Δ*(coeffs.δt*transpose(B_V))) T(-1)*(coeffs.Δ*C)
    ]
    half_BM_V⁻¹ = lazy((PNLazyMatrices.half_schur_complement, Krylov.minres), BM_V)

    A_U = sum(kron_AXB(Ut*ρp[i]*U, coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ)) for i in 1:ns.ne)
    B_U = sum(kron_AXB(Ut*∇pm[i], Ωpm[i]) for i in 1:ns.nd)
    D_U = sum(kron_AXB(Ut*∂p[i]*U, absΩp[i]) for i in 1:ns.nd)

    BM_U = [
        coeffs.Δ*(A_U + coeffs.γ*D_U) coeffs.Δ*(coeffs.δ*B_U)
        T(-1)*(coeffs.Δ*(coeffs.δt*transpose(B_U))) T(-1)*(coeffs.Δ*C)
    ]
    half_BM_U⁻¹ = lazy((PNLazyMatrices.half_schur_complement, Krylov.minres), BM_U)

    A_UV = sum(kron_AXB(Ut*ρp[i]*U, Vt*(coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ))*V) for i in 1:ns.ne)
    B_UV = sum(kron_AXB(Ut*∇pm[i], Ωpm[i]*V) for i in 1:ns.nd)
    D_UV = sum(kron_AXB(Ut*∂p[i]*U, Vt*absΩp[i]*V) for i in 1:ns.nd)

    BM_UV = [
        coeffs.Δ*(A_UV + coeffs.γ*D_UV) coeffs.Δ*(coeffs.δ*B_UV)
        T(-1)*(coeffs.Δ*(coeffs.δt*transpose(B_UV))) T(-1)*(coeffs.Δ*C)
    ]
    BM_UV⁻¹ = lazy((PNLazyMatrices.schur_complement, Krylov.minres), BM_UV)

    uBM, uhalf_BM_U⁻¹, uhalf_BM_V⁻¹, uBM_UV⁻¹ = unlazy((BM, half_BM_U⁻¹, half_BM_V⁻¹, BM_UV⁻¹), vec_size -> allocate_vec(arch, vec_size))
    rhs = allocate_vec(arch, size(BM, 1))
    tmp = allocate_vec(arch, size(BM, 1))

    return DiscreteDLRPNSystem(
        problem = pbl,
        coeffs = coeffs,
        mats = (BM=uBM, half_BM_U⁻¹=uhalf_BM_U⁻¹, half_BM_V⁻¹=uhalf_BM_V⁻¹, BM_UV⁻¹=uBM_UV⁻¹, Vt=Vt, U=U),
        rhs = rhs,
        tmp = tmp,
        max_rank=max_rank
    )
end

function step_nonadjoint!(x, system::DiscreteDLRPNSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    # update the rhs (we multiply the whole linear system with Δϵ -> "normalization")
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
    invalidate_cache!(system.mats.BM)
    # minus because we have to bring b to the right side of the equation
    assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, true)
    mul!(system.rhs, system.mats.BM, vec!(system.tmp, x), -1.0, true)

    U₀, S₀, Vt₀ = USVt(x)

    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
    # K-step (prep)
    PNLazyMatrices.resize!(system.mats.Vt, (x.rank[], :))
    PNLazyMatrices.copyto!(system.mats.Vt, Vt₀)
    # L-step (prep)
    PNLazyMatrices.resize!(system.mats.U, (:, x.rank[]))
    PNLazyMatrices.copyto!(system.mats.U, U₀)
    invalidate_cache!(system.mats.half_BM_V⁻¹) # not necessary, same ws: invalidate_cache!(system.mats.half_BM_U⁻¹)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    rhsp, rhsm = @view(system.rhs[1:nxp*nΩp]), @view(system.rhs[nxp*nΩp+1:end])

    # K-step
    rhs_K = @view(system.tmp[1:nxp*x.rank[] + nxm*nΩm])
    mul!(reshape(@view(rhs_K[1:nxp*x.rank[]]), (nxp, x.rank[])), reshape(rhsp, (nxp, nΩp)), transpose(Vt₀))
    copyto!(@view(rhs_K[nxp*x.rank[]+1:end]), rhsm)
    K₁ = allocate_vec(architecture(system.problem), nxp*x.rank[]) # TODO: allocating
    mul!(K₁, system.mats.half_BM_V⁻¹, rhs_K)
    U₁ = (qr(reshape(K₁, (nxp, x.rank[]))).Q |> mat_type(architecture(system.problem)))[1:size(U₀, 1), 1:size(U₀, 2)]
    M = transpose(U₁)*U₀

    # L-step
    rhs_Lt = @view(system.tmp[1:x.rank[]*nΩp + nxm*nΩm])
    mul!(reshape(@view(rhs_Lt[1:x.rank[]*nΩp]), (x.rank[], nΩp)), transpose(U₀), reshape(rhsp, (nxp, nΩp)))
    copyto!(@view(rhs_Lt[x.rank[]*nΩp+1:end]), rhsm)
    Lt₁ = allocate_vec(architecture(system.problem), x.rank[]*nΩp) # TODO: allocating
    mul!(Lt₁, system.mats.half_BM_U⁻¹, rhs_Lt)
    V₁ = (qr(transpose(reshape(Lt₁, (x.rank[], nΩp)))).Q |> mat_type(architecture(system.problem)))[1:size(Vt₀, 2), 1:size(Vt₀, 1)]
    N = transpose(V₁)*transpose(Vt₀)
    
    # S-step (prep)
    PNLazyMatrices.resize!(system.mats.Vt, (x.rank[], :))
    PNLazyMatrices.do_copyto!(system.mats.Vt, transpose(V₁))
    PNLazyMatrices.resize!(system.mats.U, (:, x.rank[]))
    PNLazyMatrices.do_copyto!(system.mats.U, U₁)
    invalidate_cache!(system.mats.BM_UV⁻¹)
    
    # S-step
    rhs_S = @view(system.tmp[1:x.rank[]*x.rank[] + nxm*nΩm])
    copyto!(reshape(@view(rhs_S[1:x.rank[]*x.rank[]]), (x.rank[], x.rank[])), transpose(U₁)*reshape(rhsp, (nxp, nΩp))*V₁)
    copyto!(@view(rhs_S[x.rank[]*x.rank[]+1:end]), rhsm)
    # TODO:
    S₁ = allocate_vec(architecture(system.problem), x.rank[]*x.rank[] + nxm*nΩm) # allocates, we could directly write this in S₀ and _xm
    mul!(S₁, system.mats.BM_UV⁻¹, rhs_S)

    copyto!(U₀, U₁)
    copyto!(Vt₀, transpose(V₁))
    copyto!(S₀, @view(S₁[1:x.rank[]*x.rank[]]))
    copyto!(x._xm, @view(S₁[x.rank[]*x.rank[]+1:end]))
end

# function step_adjoint!(x, system::DiscreteDLRPNSystem, rhs_ass::PNVectorAssembler, idx, Δϵ)
#     if !system.adjoint @warn "Trying to step_adjoint with system marked as nonadjoint" end
#     if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(pnsystem.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
#     # update the rhs
#     implicit_midpoint_coeffs_adjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
#     invalidate_cache!(system.BM)
#     # minus because we have to bring b to the right side of the equation
#     assemble_at!(system.rhs, rhs_ass, plus½(idx), -Δϵ, true)
#     mul!(system.rhs, transpose(system.BM), x, -1.0, true)

#     implicit_midpoint_coeffs_adjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
#     invalidate_cache!(system.BM⁻¹)
#     mul!(x, transpose(system.BM⁻¹), system.rhs)
# end

@concrete struct LowRankSolution
    _xp
    _xm

    max_rank
    rank
    nxp
    nΩp
end

function USVt(sol::LowRankSolution)
    U = reshape(@view(sol._xp[1:sol.nxp*sol.rank[]]), sol.nxp, sol.rank[])
    S = reshape(@view(sol._xp[sol.nxp*sol.rank[]+1:sol.nxp*sol.rank[]+sol.rank[]*sol.rank[]]), sol.rank[], sol.rank[])
    Vt = reshape(@view(sol._xp[sol.nxp*sol.rank[]+sol.rank[]*sol.rank[]+1:sol.nxp*sol.rank[]+sol.rank[]*sol.rank[] + sol.rank[]*sol.nΩp]), sol.rank[], sol.nΩp)
    return U, S, Vt
end

function vec!(x::AbstractVector, sol::LowRankSolution)
    Up, Sp, Vtp = USVt(sol)
    vec!(x, Up, Sp, Vtp, sol._xm)
    return x
end

function vec!(x::AbstractVector, Up, Sp, Vtp, xm)
    nxp = size(Up, 1)
    nΩp = size(Vtp, 2)
    Xp = reshape(@view(x[1:nxp*nΩp]), nxp, nΩp)
    Xp .= Up*Sp*Vtp # TODO: allocates!
    _xm = @view(x[nxp*nΩp+1:end])
    _xm .= xm
    return x
end

function pmview(sol::LowRankSolution, model::DiscretePNModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    U, S, Vt = USVt(sol)
    return U*S*Vt, reshape(sol._xm, (nxm, nΩm))
end

function allocate_solution_vector(system::DiscreteDLRPNSystem)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    max_rank = system.max_rank
    return LowRankSolution(allocate_vec(arch, nxp*max_rank + max_rank*max_rank + max_rank*nΩp), allocate_vec(arch, nxm*nΩm), max_rank, Ref(max_rank), nxp, nΩp)
end

function fillzero!(sol::LowRankSolution)
    Up, Sp, Vtp = USVt(sol)
    U, S, Vt = svd(rand(size(Up, 1), size(Vtp, 2)))
    # TODO: this is weird and allocates to work together with CUDA!
    copyto!(Up, U[:, 1:sol.rank[]])
    copyto!(Vtp, Vt[1:sol.rank[], :])
    fill!(Sp, zero(eltype(Sp)))
    fill!(sol._xm, zero(eltype(sol._xm)))
end
