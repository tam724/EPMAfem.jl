Base.@kwdef @concrete struct DiscreteDLRPNSystem4 <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp

    max_rank
end

function implicit_midpoint_dlr4(pbl::DiscretePNProblem; max_ranks=(p=20, m=20))
    if max_ranks isa Integer
        max_ranks = (p=max_ranks, m=max_ranks)
    end
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    nb = EPMAfem.n_basis(pbl)
    Δϵ = step(energy_model(pbl.model))

    Vtp = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, max_ranks.p, nb.nΩ.p), (Ref(max_ranks.p), Ref(nb.nΩ.p)))
    Vp = transpose(Vtp)
    Up = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, nb.nx.p, max_ranks.p), (Ref(nb.nx.p), Ref(max_ranks.p)))
    Utp = transpose(Up)

    Vtm = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, max_ranks.m, nb.nΩ.m), (Ref(max_ranks.m), Ref(nb.nΩ.m)))
    Vm = transpose(Vtm)
    Um = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, nb.nx.m, max_ranks.m), (Ref(nb.nx.m), Ref(max_ranks.m)))
    Utm = transpose(Um)

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

   #A(cf)   = cfs.Δ*(sum(kron_AXB(          ρp[i]    ,           Ikp(i, cf)    ) for i in 1:ns.ne) + sum(kron_AXB(          ∂p[i]    ,       cfs.γ *     absΩp[i]    ) for i in 1:ns.nd))
    Aᵥ(cf)  = cfs.Δ*(sum(kron_AXB(          ρp[i]    , cache(Vtp*Ikp(i, cf)*Vp)) for i in 1:ns.ne) + sum(kron_AXB(          ∂p[i]    , cache(cfs.γ * Vtp*absΩp[i]*Vp)) for i in 1:ns.nd))
    Aᵤ(cf)  = cfs.Δ*(sum(kron_AXB(cache(Utp*ρp[i]*Up),           Ikp(i, cf)    ) for i in 1:ns.ne) + sum(kron_AXB(cache(Utp*∂p[i]*Up),       cfs.γ *     absΩp[i]    ) for i in 1:ns.nd))
    Aᵤᵥ(cf) = cfs.Δ*(sum(kron_AXB(cache(Utp*ρp[i]*Up), cache(Vtp*Ikp(i, cf)*Vp)) for i in 1:ns.ne) + sum(kron_AXB(cache(Utp*∂p[i]*Up), cache(cfs.γ * Vtp*absΩp[i]*Vp)) for i in 1:ns.nd))
    
    # minus to symmetrize
   #C(cf)   = -(cfs.Δ*sum(kron_AXB(          ρm[i]    ,           Ikm(i, cf)    ) for i in 1:ns.ne))
    Cᵥ(cf)  = -(cfs.Δ*sum(kron_AXB(          ρm[i]    , cache(Vtm*Ikm(i, cf)*Vm)) for i in 1:ns.ne))
    Cᵤ(cf)  = -(cfs.Δ*sum(kron_AXB(cache(Utm*ρm[i]*Um),           Ikm(i, cf)    ) for i in 1:ns.ne))
    Cᵤᵥ(cf) = -(cfs.Δ*sum(kron_AXB(cache(Utm*ρm[i]*Um), cache(Vtm*Ikm(i, cf)*Vm)) for i in 1:ns.ne))

    # B   = cfs.Δ*cfs.δ*sum(kron_AXB(   ∇pm[i], Ωpm[i]  ) for i in 1:ns.nd)
    Bᵥ  = cfs.Δ*(cfs.δ*sum(kron_AXB(          ∇pm[i]    , cache(Vtm*Ωpm[i]*Vp)) for i in 1:ns.nd))
    Bᵤ  = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Utp*∇pm[i]*Um),           Ωpm[i]    ) for i in 1:ns.nd))
    Bᵤᵥ = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Utp*∇pm[i]*Um), cache(Vtm*Ωpm[i]*Vp)) for i in 1:ns.nd))

    inv_matBMᵥ = Krylov.minres([Aᵥ(cfs.mat) Bᵥ
        transpose(Bᵥ) Cᵥ(cfs.mat)])

    rhsBMᵥ = [Aᵥ(cfs.rhs) Bᵥ
        transpose(Bᵥ) Cᵥ(cfs.rhs)]

    inv_matBMᵤ = Krylov.minres([Aᵤ(cfs.mat) Bᵤ
        transpose(Bᵤ) Cᵤ(cfs.mat)])
    
    rhsBMᵤ = [Aᵤ(cfs.rhs) Bᵤ
        transpose(Bᵤ) Cᵤ(cfs.rhs)]

    inv_matBMᵤᵥ = Krylov.minres([Aᵤᵥ(cfs.mat) Bᵤᵥ
        transpose(Bᵤᵥ) Cᵤᵥ(cfs.mat)])
    
    rhsBMᵤᵥ = [Aᵤᵥ(cfs.rhs) Bᵤᵥ
        transpose(Bᵤᵥ) Cᵤᵥ(cfs.rhs)]
    

    mats = (
        Vtp = Vtp,
        Up = Up,
        Vtm = Vtm,
        Um = Um,

        inv_matBMᵥ = inv_matBMᵥ,
        rhsBMᵥ = rhsBMᵥ,

        inv_matBMᵤ = inv_matBMᵤ,
        rhsBMᵤ = rhsBMᵤ,

        inv_matBMᵤᵥ = inv_matBMᵤᵥ,
        rhsBMᵤᵥ = rhsBMᵤᵥ,
    )

    ucfs, umats = unlazy((cfs, mats), vec_size -> allocate_vec(arch, vec_size))

    rhs = (
        full = allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m),
        proj = allocate_vec(arch, max(nb.nx.p*max_ranks.p, max_ranks.p*nb.nΩ.p, max_ranks.p*max_ranks.p) + max(nb.nx.m*max_ranks.m, max_ranks.m*nb.nΩ.m, max_ranks.m*max_ranks.m))
    )

    tmp = (
        tmp1 = allocate_vec(arch, max(nb.nx.p*max_ranks.p, max_ranks.p*nb.nΩ.p, max_ranks.p*max_ranks.p) + max(nb.nx.m*max_ranks.m, max_ranks.m*nb.nΩ.m, max_ranks.m*max_ranks.m)),
        tmp2 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
        tmp3 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
    )

    return DiscreteDLRPNSystem4(
        problem = pbl,
        coeffs = ucfs,
        mats = umats,
        rhs = rhs,
        tmp = tmp,
        max_rank=max_ranks
    )
end

function step_nonadjoint!(x, system::DiscreteDLRPNSystem4, rhs_ass::PNVectorAssembler, idx, Δϵ)
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

    ((Up₀, Sp₀, Vtp₀), (Um₀, Sm₀, Vtm₀)) = USVt(x)

    CUDA.NVTX.@range "copy basis" begin
        # TODO: maybe we can skip this (this is written to the system mats before the s step..)
        # K-step (prep)
        PNLazyMatrices.resize_copyto!(system.mats.Vtp, Vtp₀)
        PNLazyMatrices.resize_copyto!(system.mats.Up, Up₀)
        PNLazyMatrices.resize_copyto!(system.mats.Vtm, Vtm₀)
        PNLazyMatrices.resize_copyto!(system.mats.Um, Um₀)
    end

    CUDA.NVTX.@range "K-step prep" begin
        rhs_K = @view(system.rhs.proj[1:nxp*x.ranks.p[]+nxm*x.ranks.m[]])
        rhs_Kp = @view(rhs_K[1:nxp*x.ranks.p[]])
        rhs_Km = @view(rhs_K[nxp*x.ranks.p[]+1:end])
        #compute û
        mul!(reshape(rhs_Kp, nxp, x.ranks.p[]), reshape(u, nxp, nΩp), transpose(Vtp₀), -1, false)
        mul!(reshape(rhs_Km, nxm, x.ranks.m[]), reshape(v, nxm, nΩm), transpose(Vtm₀), -1, false)
        # compute A₀K₀
        K₀ = @view(system.tmp.tmp1[1:nxp*x.ranks.p[] + nxm*x.ranks.m[]])
        Kp₀ = @view(K₀[1:nxp*x.ranks.p[]])
        mul!(reshape(Kp₀, nxp, x.ranks.p[]), Up₀, Sp₀)
        Km₀ = @view(K₀[nxp*x.ranks.p[]+1:end])
        mul!(reshape(Km₀, nxm, x.ranks.m[]), Um₀, Sm₀)

        mul!(rhs_K, system.mats.rhsBMᵥ, K₀, -1, true)
    end
    CUDA.NVTX.@range "K-step" begin
        K₁ = @view(system.tmp.tmp1[1:nxp*x.ranks.p[] + nxm*x.ranks.m[]])
        Kp₁ = @view(K₁[1:nxp*x.ranks.p[]])
        Km₁ = @view(K₁[nxp*x.ranks.p[]+1:end])
        mul!(K₁, system.mats.inv_matBMᵥ, rhs_K, true, false)

        Up₁ = (qr(reshape(Kp₁, (nxp, x.ranks.p[]))).Q |> mat_type(architecture(system.problem)))[1:size(Up₀, 1), 1:size(Up₀, 2)]
        Um₁ = (qr(reshape(Km₁, (nxm, x.ranks.m[]))).Q |> mat_type(architecture(system.problem)))[1:size(Um₀, 1), 1:size(Um₀, 2)]
        Mp = transpose(Up₁)*Up₀
        Mm = transpose(Um₁)*Um₀
    end

    CUDA.NVTX.@range "L-step prep" begin
        rhs_Lt = @view(system.rhs.proj[1:x.ranks.p[]*nΩp + x.ranks.m[]*nΩm])
        rhs_Ltp = @view(rhs_Lt[1:x.ranks.p[]*nΩp])
        rhs_Ltm = @view(rhs_Lt[x.ranks.p[]*nΩp+1:end])
        #compute û
        mul!(reshape(rhs_Ltp, x.ranks.p[], nΩp), transpose(Up₀), reshape(u, nxp, nΩp), -1, false)
        mul!(reshape(rhs_Ltm, x.ranks.m[], nΩm), transpose(Um₀), reshape(v, nxm, nΩm), -1, false)
        # compute A₀L₀
        L₀ = @view(system.tmp.tmp1[1:x.ranks.p[]*nΩp + x.ranks.m[]*nΩm])
        Lp₀ = @view(L₀[1:x.ranks.p[]*nΩp])
        mul!(reshape(Lp₀, x.ranks.p[], nΩp), Sp₀, Vtp₀)
        Lm₀ = @view(L₀[x.ranks.p[]*nΩp+1:end])
        mul!(reshape(Lm₀, x.ranks.m[], nΩm), Sm₀, Vtm₀)

        mul!(rhs_Lt, system.mats.rhsBMᵤ, L₀, -1, true)
    end
    CUDA.NVTX.@range "L-step" begin
        Lt₁ = @view(system.tmp.tmp1[1:x.ranks.p[]*nΩp + x.ranks.m[]*nΩm])
        Ltp₁ = @view(Lt₁[1:x.ranks.p[]*nΩp])
        Ltm₁ = @view(Lt₁[x.ranks.p[]*nΩp+1:end])
        mul!(Lt₁, system.mats.inv_matBMᵤ, rhs_Lt, true, false)

        Vp₁ = (qr(transpose(reshape(Ltp₁, (x.ranks.p[], nΩp)))).Q |> mat_type(architecture(system.problem)))[1:size(Vtp₀, 2), 1:size(Vtp₀, 1)]
        Vm₁ = (qr(transpose(reshape(Ltm₁, (x.ranks.m[], nΩm)))).Q |> mat_type(architecture(system.problem)))[1:size(Vtm₀, 2), 1:size(Vtm₀, 1)]
        Np = transpose(Vp₁)*transpose(Vtp₀)
        Nm = transpose(Vm₁)*transpose(Vtm₀)
    end

    CUDA.NVTX.@range "S-step prep" begin
        PNLazyMatrices.resize_copyto!(system.mats.Vtp, transpose(Vp₁))
        PNLazyMatrices.resize_copyto!(system.mats.Vtm, transpose(Vm₁))
        PNLazyMatrices.resize_copyto!(system.mats.Up, Up₁)
        PNLazyMatrices.resize_copyto!(system.mats.Um, Um₁)

        rhs_S = @view(system.rhs.proj[1:x.ranks.p[]*x.ranks.p[] + x.ranks.m[]*x.ranks.m[]])
        rhs_Sp = @view(rhs_S[1:x.ranks.p[]*x.ranks.p[]])
        rhs_Sm = @view(rhs_S[x.ranks.p[]*x.ranks.p[]+1:end])
        #compute û
        reshape(rhs_Sp, (x.ranks.p[], x.ranks.p[])) .= -transpose(Up₁)*reshape(u, nxp, nΩp)*Vp₁
        reshape(rhs_Sm, (x.ranks.m[], x.ranks.m[])) .= -transpose(Um₁)*reshape(v, nxm, nΩm)*Vm₁
        # compute A₀Ŝ₀
        S₀_hat = @view(system.tmp.tmp1[1:x.ranks.p[]*x.ranks.p[] + x.ranks.m[]*x.ranks.m[]])
        Sp₀_hat = @view(S₀_hat[1:x.ranks.p[]*x.ranks.p[]])
        Sm₀_hat = @view(S₀_hat[x.ranks.p[]*x.ranks.p[]+1:end])
        reshape(Sp₀_hat, x.ranks.p[], x.ranks.p[]) .= Mp * Sp₀ * transpose(Np)
        reshape(Sm₀_hat, x.ranks.m[], x.ranks.m[]) .= Mm * Sm₀ * transpose(Nm)

        mul!(rhs_S, system.mats.rhsBMᵤᵥ, S₀_hat, -1, true)
    end    
    CUDA.NVTX.@range "S-step" begin
        S₁ = @view(system.tmp.tmp1[1:x.ranks.p[]*x.ranks.p[] + x.ranks.m[]*x.ranks.m[]])
        Sp₁ = @view(S₁[1:x.ranks.p[]*x.ranks.p[]])
        Sm₁ = @view(S₁[x.ranks.p[]*x.ranks.p[]+1:end])

        mul!(S₁, system.mats.inv_matBMᵤᵥ, rhs_S, true, false)
    end
    CUDA.NVTX.@range "finalize schur" begin
        copyto!(Up₀, Up₁)
        copyto!(Um₀, Um₁)
        copyto!(Vtp₀, transpose(Vp₁))
        copyto!(Vtm₀, transpose(Vm₁))
        copyto!(Sp₀, Sp₁)
        copyto!(Sm₀, Sm₁)
    end
end

function allocate_solution_vector(system::DiscreteDLRPNSystem4)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    max_ranks = system.max_rank
    return LowwRankSolution(
        allocate_vec(arch, nxp*max_ranks.p + max_ranks.p*max_ranks.p + max_ranks.p*nΩp),
        allocate_vec(arch, nxm*max_ranks.m + max_ranks.m*max_ranks.m + max_ranks.m*nΩm),
        max_ranks, 
        (p=Ref(max_ranks.p), m=Ref(max_ranks.m)),
        n_basis(system.problem)
    )
end

@concrete struct LowwRankSolution
    _xp
    _xm

    max_ranks
    ranks
    nb
end

function _USVt(x::AbstractVector, rank, nL, nR)
    U = reshape(@view(x[1:nL*rank]), nL, rank)
    S = reshape(@view(x[nL*rank+1:nL*rank+rank*rank]), rank, rank)
    Vt = reshape(@view(x[nL*rank+rank*rank+1:nL*rank+rank*rank + rank*nR]), rank, nR)
    return U, S, Vt
end

function USVt(sol::LowwRankSolution)
    Up, Sp, Vtp = _USVt(sol._xp, sol.ranks.p[], sol.nb.nx.p, sol.nb.nΩ.p)
    Um, Sm, Vtm = _USVt(sol._xm, sol.ranks.m[], sol.nb.nx.m, sol.nb.nΩ.m)
    return (p=(Up, Sp, Vtp), m=(Um, Sm, Vtm))
end

function pview(sol::LowwRankSolution, model::DiscretePNModel)
    U, S, Vt = USVt(sol).p
    return U*S*Vt
end

function mview(sol::LowwRankSolution, model::DiscretePNModel)
    U, S, Vt = USVt(sol).m
    return U*S*Vt
end

function pmview(sol::LowwRankSolution, model::DiscretePNModel)
    ((Up, Sp, Vtp), (Um, Sm, Vtm)) = USVt(sol)
    return Up*Sp*Vtp, Um*Sm*Vtm
end

function fillzero!(sol::LowwRankSolution)
    ((Up, Sp, Vtp), (Um, Sm, Vtm)) = USVt(sol)
    Up_, _, Vtp_ = svd(rand(size(Up, 1), size(Vtp, 2)))
    # TODO: this is weird and allocates to work together with CUDA!
    copyto!(Up, Up_[:, 1:sol.ranks.p[]])
    copyto!(Vtp, Vtp_[1:sol.ranks.p[], :])
    fill!(Sp, zero(eltype(Sp)))

    Um_, _, Vtm_ = svd(rand(size(Um, 1), size(Vtm, 2)))
    # TODO: this is weird and allocates to work together with CUDA!
    copyto!(Um, Um_[:, 1:sol.ranks.m[]])
    copyto!(Vtm, Vtm_[1:sol.ranks.m[], :])
    fill!(Sm, zero(eltype(Sm)))
end
