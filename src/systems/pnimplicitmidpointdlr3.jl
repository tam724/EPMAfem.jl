Base.@kwdef @concrete struct DiscreteDLRPNSystem3 <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp

    max_rank
end

function implicit_midpoint_dlr3(pbl::DiscretePNProblem; max_rank=20)
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    nb = EPMAfem.n_basis(pbl)
    Δϵ = step(energy_model(pbl.model))

    Vt = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, max_rank, nb.nΩ.p), (Ref(max_rank), Ref(nb.nΩ.p)))
    V = transpose(Vt)
    U = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, nb.nx.p, max_rank), (Ref(nb.nx.p), Ref(max_rank)))
    Ut = transpose(U)

    cfs = (
        Δ=LazyScalar(T(Δϵ)),
        γ=LazyScalar(T(0.5)),
        δ=LazyScalar(T(-0.5)),
        mat = (a = [LazyScalar(zero(T)) for _ in 1:ns.ne], c = [[LazyScalar(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne]),
        rhs = (a = [LazyScalar(zero(T)) for _ in 1:ns.ne], c = [[LazyScalar(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne])
        )

    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(pbl)

    Ikp(i, cf) = materialize(cf.a[i]*Ip + sum(cf.c[i][j]*kp[i][j] for j in 1:ns.nσ))
    Ikm(i, cf) = materialize(cf.a[i]*Im + sum(cf.c[i][j]*km[i][j] for j in 1:ns.nσ))

    # A(cf)   = cfs.Δ*(sum(kron_AXB(   ρp[i],      Ikp(i, cf)  ) for i in 1:ns.ne) + sum(kron_AXB(   ∂p[i],   cfs.γ *    absΩp[i]  ) for i in 1:ns.nd))
    Aᵥ(cf)  = cfs.Δ*(sum(kron_AXB(         ρp[i]   , cache(Vt*Ikp(i, cf)*V)) for i in 1:ns.ne) + sum(kron_AXB(         ∂p[i]   , cache(cfs.γ * Vt*absΩp[i]*V)) for i in 1:ns.nd))
    Aᵤ(cf)  = cfs.Δ*(sum(kron_AXB(cache(Ut*ρp[i]*U),          Ikp(i, cf)   ) for i in 1:ns.ne) + sum(kron_AXB(cache(Ut*∂p[i]*U),       cfs.γ *    absΩp[i]   ) for i in 1:ns.nd))
    Aᵤᵥ(cf) = cfs.Δ*(sum(kron_AXB(cache(Ut*ρp[i]*U), cache(Vt*Ikp(i, cf)*V)) for i in 1:ns.ne) + sum(kron_AXB(cache(Ut*∂p[i]*U), cache(cfs.γ * Vt*absΩp[i]*V)) for i in 1:ns.nd))
    
    # minus to symmetrize
    C(cf) = -(cfs.Δ*sum(kron_AXB(ρm[i], Ikm(i, cf)) for i in 1:ns.ne))

    # B   = cfs.Δ*cfs.δ*sum(kron_AXB(   ∇pm[i], Ωpm[i]  ) for i in 1:ns.nd)
    Bᵥ  = cfs.Δ*(cfs.δ*sum(kron_AXB(         ∇pm[i] , cache(Ωpm[i]*V)) for i in 1:ns.nd))
    Bᵤ  = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Ut*∇pm[i]),       Ωpm[i]   ) for i in 1:ns.nd))
    Bᵤᵥ = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Ut*∇pm[i]), cache(Ωpm[i]*V)) for i in 1:ns.nd))

    rhsC = C(cfs.rhs)
    matC⁻¹ = cache(LinearAlgebra.inv!(C(cfs.mat)))
    inv_AᵥmBᵥCBᵥᵀ    = Krylov.minres( Aᵥ(cfs.mat) - Bᵥ  * matC⁻¹ * transpose(Bᵥ )) # TODO: maybe there is a way to make this term cheap?
    inv_AᵤmBᵤCBᵤᵀ    = Krylov.minres( Aᵤ(cfs.mat) - Bᵤ  * matC⁻¹ * transpose(Bᵤ ))
    inv_AᵤᵥmBᵤᵥCBᵤᵥᵀ = Krylov.minres(Aᵤᵥ(cfs.mat) - Bᵤᵥ * matC⁻¹ * transpose(Bᵤᵥ))

    mats = (
        Vt = Vt,
        U = U,
        rhsC = rhsC,
        matC⁻¹ = matC⁻¹,
        rhsAᵥ = Aᵥ(cfs.rhs),
        Bᵥ = Bᵥ,
        inv_AᵥmBᵥCBᵥᵀ = inv_AᵥmBᵥCBᵥᵀ,

        rhsAᵤ = Aᵤ(cfs.rhs),
        Bᵤ = Bᵤ,
        inv_AᵤmBᵤCBᵤᵀ = inv_AᵤmBᵤCBᵤᵀ,

        rhsAᵤᵥ = Aᵤᵥ(cfs.rhs),
        Bᵤᵥ = Bᵤᵥ,
        inv_AᵤᵥmBᵤᵥCBᵤᵥᵀ = inv_AᵤᵥmBᵤᵥCBᵤᵥᵀ,
    )

    ucfs, umats = unlazy((cfs, mats), vec_size -> allocate_vec(arch, vec_size))

    rhs = (
        full = allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m),
        proj = allocate_vec(arch, max(nb.nx.p*max_rank, max_rank*nb.nΩ.p, max_rank*max_rank))
    )

    tmp = (
        tmp1 = allocate_vec(arch, max(nb.nx.p*max_rank, max_rank*nb.nΩ.p, max_rank*max_rank)),
        tmp2 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
        tmp3 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
    )

    return DiscreteDLRPNSystem3(
        problem = pbl,
        coeffs = ucfs,
        mats = umats,
        rhs = rhs,
        tmp = tmp,
        max_rank=max_rank
    )
end

function step_nonadjoint!(x, system::DiscreteDLRPNSystem3, rhs_ass::PNVectorAssembler, idx, Δϵ)
    @show idx
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs.rhs, system.problem, idx, Δϵ)
    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs.mat, system.problem, idx, Δϵ)

    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)

    CUDA.NVTX.@range "rhs assembly" begin
        assemble_at!(system.rhs.full, rhs_ass, minus½(idx), Δϵ, true)
        u = @view(system.rhs.full[1:nxp*nΩp])
        v = @view(system.rhs.full[nxp*nΩp+1:end])
    end

    CUDA.NVTX.@range "prep" begin
        copyto!(system.tmp.tmp3, v) #TODO: check if we can simply use the rhs for that!
        mul!(system.tmp.tmp3, system.mats.rhsC, x._xm)
    end

    U₀, S₀, Vt₀ = USVt(x)

    CUDA.NVTX.@range "copy basis" begin
        # TODO: maybe we can skip this (this is written to the system mats before the s step..)
        # K-step (prep)
        PNLazyMatrices.resize_copyto!(system.mats.Vt, Vt₀)
        # L-step (prep)
        PNLazyMatrices.resize_copyto!(system.mats.U, U₀)
    end

    CUDA.NVTX.@range "K-step prep" begin
        rhs_K = @view(system.rhs.proj[1:nxp*x.rank[]])
        #compute û
        mul!(reshape(rhs_K, nxp, x.rank[]), reshape(u, nxp, nΩp), transpose(Vt₀), -1, false)
        # compute A₀K₀
        K₀ = @view(system.tmp.tmp1[1:nxp*x.rank[]])
        mul!(reshape(K₀, nxp, x.rank[]), U₀, S₀)

        mul!(rhs_K, system.mats.rhsAᵥ, K₀, -1, true)

        #compute
        copyto!(system.tmp.tmp2, system.tmp.tmp3) # tmp3 = Cᵢ*yᵢ + v
        mul!(system.tmp.tmp2, transpose(system.mats.Bᵥ), K₀, true, true)
        mul!(system.tmp.tmp2, system.mats.matC⁻¹, system.tmp.tmp2, true, false)
        system.tmp.tmp2 .= x._xm .- system.tmp.tmp2

        mul!(rhs_K, system.mats.Bᵥ, system.tmp.tmp2, -1, true)
    end
    CUDA.NVTX.@range "K-step" begin
        K₁ = @view(system.tmp.tmp1[1:nxp*x.rank[]])
        mul!(K₁, system.mats.inv_AᵥmBᵥCBᵥᵀ, rhs_K, true, false)
        U₁ = (qr(reshape(K₁, (nxp, x.rank[]))).Q |> mat_type(architecture(system.problem)))[1:size(U₀, 1), 1:size(U₀, 2)]
        M = transpose(U₁)*U₀
    end

    CUDA.NVTX.@range "L-step prep" begin
        rhs_Lt = @view(system.rhs.proj[1:x.rank[]*nΩp])
        #compute û
        mul!(reshape(rhs_Lt, x.rank[], nΩp), transpose(U₀), reshape(u, nxp, nΩp), -1, false)
        # compute A₀L₀
        L₀ = @view(system.tmp.tmp1[1:x.rank[]*nΩp])
        mul!(reshape(L₀, x.rank[], nΩp), S₀, Vt₀)

        mul!(rhs_Lt, system.mats.rhsAᵤ, L₀, -1, true)

        #compute
        copyto!(system.tmp.tmp2, system.tmp.tmp3) # tmp3 = Cᵢ*yᵢ + v
        mul!(system.tmp.tmp2, transpose(system.mats.Bᵤ), L₀, true, true)
        mul!(system.tmp.tmp2, system.mats.matC⁻¹, system.tmp.tmp2, true, false)
        system.tmp.tmp2 .= x._xm .- system.tmp.tmp2

        mul!(rhs_Lt, system.mats.Bᵤ, system.tmp.tmp2, -1, true)
    end
    CUDA.NVTX.@range "L-step" begin
        Lt₁ = @view(system.tmp.tmp1[1:x.rank[]*nΩp])
        mul!(Lt₁, system.mats.inv_AᵤmBᵤCBᵤᵀ, rhs_Lt, true, false)
        V₁ = (qr(transpose(reshape(Lt₁, (x.rank[], nΩp)))).Q |> mat_type(architecture(system.problem)))[1:size(Vt₀, 2), 1:size(Vt₀, 1)]
        N = transpose(V₁)*transpose(Vt₀)
    end

    CUDA.NVTX.@range "S-step prep" begin
        PNLazyMatrices.resize_copyto!(system.mats.Vt, transpose(V₁))
        PNLazyMatrices.resize_copyto!(system.mats.U, U₁)

        rhs_S = @view(system.rhs.proj[1:x.rank[]*x.rank[]])
        #compute û
        reshape(rhs_S, (x.rank[], x.rank[])) .= -transpose(U₁)*reshape(u, nxp, nΩp)*V₁
        # compute A₀Ŝ₀
        S₀_hat = @view(system.tmp.tmp1[1:x.rank[]*x.rank[]])
        reshape(S₀_hat, x.rank[], x.rank[]) .= M * S₀ * transpose(N)

        mul!(rhs_S, system.mats.rhsAᵤᵥ, S₀_hat, -1, true)

        #compute
        copyto!(system.tmp.tmp2, system.tmp.tmp3) # tmp3 = Cᵢ*yᵢ + v
        mul!(system.tmp.tmp2, transpose(system.mats.Bᵤᵥ), S₀_hat, true, true)
        copyto!(system.tmp.tmp3, system.tmp.tmp2) # save for schur complement
        mul!(system.tmp.tmp2, system.mats.matC⁻¹, system.tmp.tmp2, true, false) # TODO: save this here for the rhs of the schur complement?
        system.tmp.tmp2 .= x._xm .- system.tmp.tmp2

        mul!(rhs_S, system.mats.Bᵤᵥ, system.tmp.tmp2, -1, true)
    end    
    CUDA.NVTX.@range "S-step" begin
        S₁ = @view(system.tmp.tmp1[1:x.rank[]*x.rank[]])
        mul!(S₁, system.mats.inv_AᵤᵥmBᵤᵥCBᵤᵥᵀ, rhs_S, true, false)
    end
    CUDA.NVTX.@range "finalize schur" begin
        mul!(system.tmp.tmp3, transpose(system.mats.Bᵤᵥ), S₁, -1, -1)
        mul!(x._xm, system.mats.matC⁻¹, system.tmp.tmp3, true, false)

        copyto!(U₀, U₁)
        copyto!(Vt₀, transpose(V₁))
        copyto!(S₀, @view(S₁[1:x.rank[]*x.rank[]]))
    end
end

function allocate_solution_vector(system::DiscreteDLRPNSystem3)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    max_rank = system.max_rank
    return LowRankSolution(allocate_vec(arch, nxp*max_rank + max_rank*max_rank + max_rank*nΩp), allocate_vec(arch, nxm*nΩm), max_rank, Ref(max_rank), nxp, nΩp)
end
