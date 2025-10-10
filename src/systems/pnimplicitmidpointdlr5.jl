Base.@kwdef @concrete struct DiscreteDLRPNSystem5{ADAPTIVE} <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp

    max_ranks
    tolerance::ADAPTIVE
    basis_augmentation
    conserved_quantities
end

function Base.adjoint(A::DiscreteDLRPNSystem5{ADAPTIVE}) where ADAPTIVE
    return DiscreteDLRPNSystem5(adjoint=!A.adjoint, problem=A.problem, coeffs=A.coeffs, mats=A.mats, rhs=A.rhs, tmp=A.tmp, max_ranks=A.max_ranks, tolerance=A.tolerance, basis_augmentation=A.basis_augmentation, conserved_quantities=A.conserved_quantities)
end

adaptive(::DiscreteDLRPNSystem5{Nothing}) = false
adaptive(::DiscreteDLRPNSystem5{<:Real}) = true

using EPMAfem.PNLazyMatrices: only_unique

function implicit_midpoint_dlr5(pbl::DiscretePNProblem; solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=nothing, basis_augmentation=nothing, conserved_quantities=nothing)
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    nb = EPMAfem.n_basis(pbl)
    Δϵ = step(energy_model(pbl.model))

    if max_ranks isa Integer
        max_ranks = (p=max_ranks, m=max_ranks)
    end
    if max_ranks.p > fld(nb.nΩ.p, 2) || max_ranks.m > fld(nb.nΩ.m, 2) # trim the number_of_ranks to nΩ / 2
        max_ranks = (p=min(max_ranks.p, fld(nb.nΩ.p, 2)), m=min(max_ranks.m, fld(nb.nΩ.m, 2)))
        @warn "Trimming the number of max ranks to nΩ/2: $(max_ranks)"
    end

    n_bas_aug_p = isnothing(basis_augmentation) ? 0 : only_unique(size(basis_augmentation.p.U, 2), size(basis_augmentation.p.V, 2))
    n_bas_aug_m = isnothing(basis_augmentation) ? 0 : only_unique(size(basis_augmentation.m.U, 2), size(basis_augmentation.m.V, 2))

    max_aug_ranks = (p=2max_ranks.p+n_bas_aug_p, m=2max_ranks.m+n_bas_aug_m)

    Vtp = PNLazyMatrices.LazyResizeMatrix(Ref(allocate_vec(arch, 0)), (max_aug_ranks.p, nb.nΩ.p), (Ref(max_aug_ranks.p), Ref(nb.nΩ.p)))
    Vp = transpose(Vtp)
    Up = PNLazyMatrices.LazyResizeMatrix(Ref(allocate_vec(arch, 0)), (nb.nx.p, max_aug_ranks.p), (Ref(nb.nx.p), Ref(max_aug_ranks.p)))
    Utp = transpose(Up)

    Vtm = PNLazyMatrices.LazyResizeMatrix(Ref(allocate_vec(arch, 0)), (max_aug_ranks.m, nb.nΩ.m), (Ref(max_aug_ranks.m), Ref(nb.nΩ.m)))
    Vm = transpose(Vtm)
    Um = PNLazyMatrices.LazyResizeMatrix(Ref(allocate_vec(arch, 0)), (nb.nx.m, max_aug_ranks.m), (Ref(nb.nx.m), Ref(max_aug_ranks.m)))
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

    A(cf)   = cfs.Δ*(sum(kron_AXB(          ρp[i]    ,           Ikp(i, cf)    ) for i in 1:ns.ne) + sum(kron_AXB(          ∂p[i]    ,       cfs.γ *     absΩp[i]    ) for i in 1:ns.nd))
    Aᵥ(cf)  = cfs.Δ*(sum(kron_AXB(          ρp[i]    , cache(Vtp*Ikp(i, cf)*Vp)) for i in 1:ns.ne) + sum(kron_AXB(          ∂p[i]    , cache(cfs.γ * Vtp*absΩp[i]*Vp)) for i in 1:ns.nd))
    Aᵤ(cf)  = cfs.Δ*(sum(kron_AXB(cache(Utp*ρp[i]*Up),           Ikp(i, cf)    ) for i in 1:ns.ne) + sum(kron_AXB(cache(Utp*∂p[i]*Up),       cfs.γ *     absΩp[i]    ) for i in 1:ns.nd))
    Aᵤᵥ(cf) = cfs.Δ*(sum(kron_AXB(cache(Utp*ρp[i]*Up), cache(Vtp*Ikp(i, cf)*Vp)) for i in 1:ns.ne) + sum(kron_AXB(cache(Utp*∂p[i]*Up), cache(cfs.γ * Vtp*absΩp[i]*Vp)) for i in 1:ns.nd))
    
    # minus to symmetrize
    C(cf)   = -(cfs.Δ*sum(kron_AXB(          ρm[i]    ,           Ikm(i, cf)    ) for i in 1:ns.ne))
    Cᵥ(cf)  = -(cfs.Δ*sum(kron_AXB(          ρm[i]    , cache(Vtm*Ikm(i, cf)*Vm)) for i in 1:ns.ne))
    Cᵤ(cf)  = -(cfs.Δ*sum(kron_AXB(cache(Utm*ρm[i]*Um),           Ikm(i, cf)    ) for i in 1:ns.ne))
    Cᵤᵥ(cf) = -(cfs.Δ*sum(kron_AXB(cache(Utm*ρm[i]*Um), cache(Vtm*Ikm(i, cf)*Vm)) for i in 1:ns.ne))

    B   = cfs.Δ*(cfs.δ*sum(kron_AXB(          ∇pm[i]    ,           Ωpm[i]    ) for i in 1:ns.nd))
    Bᵥ  = cfs.Δ*(cfs.δ*sum(kron_AXB(          ∇pm[i]    , cache(Vtm*Ωpm[i]*Vp)) for i in 1:ns.nd))
    Bᵤ  = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Utp*∇pm[i]*Um),           Ωpm[i]    ) for i in 1:ns.nd))
    Bᵤᵥ = cfs.Δ*(cfs.δ*sum(kron_AXB(cache(Utp*∇pm[i]*Um), cache(Vtm*Ωpm[i]*Vp)) for i in 1:ns.nd))

    inv_matBM = solver([A(cfs.mat) B
        transpose(B) C(cfs.mat)])

    rhsBM = [A(cfs.rhs) B
        transpose(B) C(cfs.rhs)]

    inv_matBMᵥ = solver([Aᵥ(cfs.mat) Bᵥ
        transpose(Bᵥ) Cᵥ(cfs.mat)])

    rhsBMᵥ = [Aᵥ(cfs.rhs) Bᵥ
        transpose(Bᵥ) Cᵥ(cfs.rhs)]

    inv_matBMᵤ = solver([Aᵤ(cfs.mat) Bᵤ
        transpose(Bᵤ) Cᵤ(cfs.mat)])
    
    rhsBMᵤ = [Aᵤ(cfs.rhs) Bᵤ
        transpose(Bᵤ) Cᵤ(cfs.rhs)]

    inv_matBMᵤᵥ = solver([Aᵤᵥ(cfs.mat) Bᵤᵥ
        transpose(Bᵤᵥ) Cᵤᵥ(cfs.mat)])
    
    rhsBMᵤᵥ = [Aᵤᵥ(cfs.rhs) Bᵤᵥ
        transpose(Bᵤᵥ) Cᵤᵥ(cfs.rhs)]
    

    mats = (
        Vtp = Vtp,
        Up = Up,
        Vtm = Vtm,
        Um = Um,

        inv_matBM = inv_matBM,
        rhsBM = rhsBM,

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
        proj = allocate_vec(arch, max(nb.nx.p*max_aug_ranks.p, max_aug_ranks.p*nb.nΩ.p, max_aug_ranks.p*max_aug_ranks.p) + max(nb.nx.m*max_aug_ranks.m, max_aug_ranks.m*nb.nΩ.m, max_aug_ranks.m*max_aug_ranks.m))
    )

    tmp = (
        tmp1 = allocate_vec(arch, max(nb.nx.p*max_aug_ranks.p, max_aug_ranks.p*nb.nΩ.p, max_aug_ranks.p*max_aug_ranks.p) + max(nb.nx.m*max_aug_ranks.m, max_aug_ranks.m*nb.nΩ.m, max_aug_ranks.m*max_aug_ranks.m)),
        tmp2 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
        tmp3 = allocate_vec(arch, nb.nx.m*nb.nΩ.m),
    )

    return DiscreteDLRPNSystem5(
        problem = pbl,
        coeffs = ucfs,
        mats = umats,
        rhs = rhs,
        tmp = tmp,
        max_ranks=max_ranks,
        tolerance=tolerance,
        basis_augmentation=basis_augmentation,
        conserved_quantities=conserved_quantities
    )
end

function step_nonadjoint!(x, system::DiscreteDLRPNSystem5, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs.rhs, system.problem, idx, Δϵ)
    implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs.mat, system.problem, idx, Δϵ)
    _step!(x, system, rhs_ass, minus½(idx), Δϵ)
end

function step_adjoint!(x, system::DiscreteDLRPNSystem5, rhs_ass::PNVectorAssembler, idx, Δϵ)
    if !system.adjoint @warn "Trying to step_adjoint with system marked as nonadjoint" end
    if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
    implicit_midpoint_coeffs_adjoint_rhs!(system.coeffs.rhs, system.problem, idx, Δϵ)
    implicit_midpoint_coeffs_adjoint_mat!(system.coeffs.mat, system.problem, idx, Δϵ)
    _step!(x, system, rhs_ass, plus½(idx), Δϵ)
end

function _compute_new_ranks(system::DiscreteDLRPNSystem5{<:Real}, Sp, Sm)
    Sp, Sm = collect(Sp), collect(Sm)
    #p 
    Σσ²p = 0.0
    rp = length(Sp)
    norm_Sp² = Sp[1] # sum(Sp.^2)
    while (sqrt(Σσ²p + Sp[rp]^2) < system.tolerance^2*norm_Sp²) || (sqrt(Σσ²p + Sp[rp]^2) < 1e-12) && rp > 1
        Σσ²p += Sp[rp]^2
        rp = rp - 1
    end
    #m  
    Σσ²m = 0.0
    rm = length(Sm)
    norm_Sm² = Sm[1] # sum(Sm.^2)
    while (sqrt(Σσ²m + Sm[rm]^2) < system.tolerance^2*norm_Sm²) || (sqrt(Σσ²m + Sm[rm]^2) < 1e-12) && rm > 1
        Σσ²m += Sm[rm]^2
        rm = rm - 1
    end
    return (p=min(system.max_ranks.p, rp), m=min(system.max_ranks.m, rm))
end

_compute_new_ranks(system::DiscreteDLRPNSystem5{<:Nothing}, _, _) = system.max_ranks

function _orthonormalize(bases...; tol=1e-15)
    Q_combined = bases[1] # retain the first basis
    for i in 2:length(bases)
        Qi = bases[i]
        
        # remove components in span(Q_combined)
        Qi_proj = Qi - Q_combined * (Q_combined' * Qi)
        # prthonormalize remaining directions
        Qi_orth, R = qr(Qi_proj)
        # truncate
        diagR = abs.(diag(R))
        Qi_clean = Matrix(Qi_orth)[:, diagR .> tol]
        
        Q_combined = hcat(Q_combined, Qi_clean)
    end

    return Matrix(qr(Q_combined).Q)
end

function _step!(x, system::DiscreteDLRPNSystem5, rhs_ass::PNVectorAssembler, idx, Δϵ)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)

    CUDA.NVTX.@range "rhs assembly" begin
        assemble_at!(system.rhs.full, rhs_ass, idx, Δϵ, true)
        u = @view(system.rhs.full[1:nxp*nΩp])
        v = @view(system.rhs.full[nxp*nΩp+1:end])
    end

    ((Up₀, Sp₀, Vtp₀), (Um₀, Sm₀, Vtm₀)) = USVt(x)
    ranks = (p=x.ranks.p[], m=x.ranks.m[])
    
    CUDA.NVTX.@range "copy basis" begin
        # TODO: maybe we can skip this (this is written to the system mats before the s step..)
        # K-step (prep)
        PNLazyMatrices.set!(system.mats.Vtp, vec(Vtp₀), size(Vtp₀))
        PNLazyMatrices.set!(system.mats.Vtm, vec(Vtm₀), size(Vtm₀))
        PNLazyMatrices.set!(system.mats.Up, vec(Up₀), size(Up₀))
        PNLazyMatrices.set!(system.mats.Um, vec(Um₀), size(Um₀))
    end

    CUDA.NVTX.@range "K-step prep" begin
        rhs_K = @view(system.rhs.proj[1:nxp*ranks.p+nxm*ranks.m])
        rhs_Kp = @view(rhs_K[1:nxp*ranks.p])
        rhs_Km = @view(rhs_K[nxp*ranks.p+1:end])
        #compute û
        mul!(reshape(rhs_Kp, nxp, ranks.p), reshape(u, nxp, nΩp), transpose(Vtp₀), -1, false)
        mul!(reshape(rhs_Km, nxm, ranks.m), reshape(v, nxm, nΩm), transpose(Vtm₀), -1, false)
        # compute A₀K₀
        K₀ = @view(system.tmp.tmp1[1:nxp*ranks.p + nxm*ranks.m])
        Kp₀ = @view(K₀[1:nxp*ranks.p])
        mul!(reshape(Kp₀, nxp, ranks.p), Up₀, Sp₀)
        Km₀ = @view(K₀[nxp*ranks.p+1:end])
        mul!(reshape(Km₀, nxm, ranks.m), Um₀, Sm₀)

        mul!(rhs_K, system.mats.rhsBMᵥ, K₀, -1, true)
    end
    CUDA.NVTX.@range "K-step" begin
        K₁ = @view(system.tmp.tmp1[1:nxp*ranks.p + nxm*ranks.m])
        Kp₁ = @view(K₁[1:nxp*ranks.p])
        Km₁ = @view(K₁[nxp*ranks.p+1:end])
        mul!(K₁, system.mats.inv_matBMᵥ, rhs_K, true, false)

        if !isnothing(system.basis_augmentation)
            Uphat = _orthonormalize(system.basis_augmentation.p.U, reshape(Kp₁, (nxp, ranks.p)), Up₀)
        else
            Uphat = _orthonormalize(reshape(Kp₁, (nxp, ranks.p)), Up₀)
        end
        aug_ranks_p_u = size(Uphat, 2)
        @show abs.(transpose(Uphat) * Uphat - I) |> maximum

        if !isnothing(system.basis_augmentation)
            Umhat = _orthonormalize(system.basis_augmentation.m.U, reshape(Km₁, (nxm, ranks.m)), Um₀)
        else
            Umhat = _orthonormalize(reshape(Km₁, (nxm, ranks.m)), Um₀)
        end
        aug_ranks_m_u = size(Umhat, 2)
        @show abs.(transpose(Umhat) * Umhat - I) |> maximum


        Mp = transpose(Uphat)*Up₀
        Mm = transpose(Umhat)*Um₀
    end

    CUDA.NVTX.@range "L-step prep" begin
        rhs_Lt = @view(system.rhs.proj[1:ranks.p*nΩp + ranks.m*nΩm])
        rhs_Ltp = @view(rhs_Lt[1:ranks.p*nΩp])
        rhs_Ltm = @view(rhs_Lt[ranks.p*nΩp+1:end])
        #compute û
        mul!(reshape(rhs_Ltp, ranks.p, nΩp), transpose(Up₀), reshape(u, nxp, nΩp), -1, false)
        mul!(reshape(rhs_Ltm, ranks.m, nΩm), transpose(Um₀), reshape(v, nxm, nΩm), -1, false)
        # compute A₀L₀
        L₀ = @view(system.tmp.tmp1[1:ranks.p*nΩp + ranks.m*nΩm])
        Lp₀ = @view(L₀[1:ranks.p*nΩp])
        mul!(reshape(Lp₀, ranks.p, nΩp), Sp₀, Vtp₀)
        Lm₀ = @view(L₀[ranks.p*nΩp+1:end])
        mul!(reshape(Lm₀, ranks.m, nΩm), Sm₀, Vtm₀)

        mul!(rhs_Lt, system.mats.rhsBMᵤ, L₀, -1, true)
    end
    CUDA.NVTX.@range "L-step" begin
        Lt₁ = @view(system.tmp.tmp1[1:ranks.p*nΩp + ranks.m*nΩm])
        Ltp₁ = @view(Lt₁[1:ranks.p*nΩp])
        Ltm₁ = @view(Lt₁[ranks.p*nΩp+1:end])
        mul!(Lt₁, system.mats.inv_matBMᵤ, rhs_Lt, true, false)
        
        if !isnothing(system.basis_augmentation)
            Vphat = _orthonormalize(system.basis_augmentation.p.V, transpose(reshape(Ltp₁, (ranks.p, nΩp))), transpose(Vtp₀))
        else
            Vphat = _orthonormalize(transpose(reshape(Ltp₁, (ranks.p, nΩp))), transpose(Vtp₀))
        end
        aug_ranks_p_v = size(Vphat, 2)
        @show abs.(transpose(Vphat) * Vphat - I) |> maximum

        if !isnothing(system.basis_augmentation)
            Vmhat = _orthonormalize(system.basis_augmentation.m.V, transpose(reshape(Ltm₁, (ranks.m, nΩm))), transpose(Vtm₀))
        else
            Vmhat = _orthonormalize(transpose(reshape(Ltm₁, (ranks.m, nΩm))), transpose(Vtm₀))
        end
        aug_ranks_m_v = size(Vmhat, 2)
        @show abs.(transpose(Vmhat) * Vmhat - I) |> maximum

        Np = transpose(Vphat)*transpose(Vtp₀)
        Nm = transpose(Vmhat)*transpose(Vtm₀)
    end

    CUDA.NVTX.@range "S-step prep" begin
        Vtphat_ = transpose(Vphat) |> mat_type(architecture(system.problem))
        Vtmhat_ = transpose(Vmhat) |> mat_type(architecture(system.problem))
        PNLazyMatrices.set!(system.mats.Vtp, vec(Vtphat_), size(Vtphat_))
        PNLazyMatrices.set!(system.mats.Vtm, vec(Vtmhat_), size(Vtmhat_))
        PNLazyMatrices.set!(system.mats.Up, vec(Uphat), size(Uphat))
        PNLazyMatrices.set!(system.mats.Um, vec(Umhat), size(Umhat))

        rhs_S = @view(system.rhs.proj[1:aug_ranks_p_u*aug_ranks_p_v + aug_ranks_m_u*aug_ranks_m_v])
        rhs_Sp = @view(rhs_S[1:aug_ranks_p_u*aug_ranks_p_v])
        rhs_Sm = @view(rhs_S[aug_ranks_p_u*aug_ranks_p_v+1:end])
        #compute û
        reshape(rhs_Sp, (aug_ranks_p_u, aug_ranks_p_v)) .= -transpose(Uphat)*reshape(u, nxp, nΩp)*Vphat
        reshape(rhs_Sm, (aug_ranks_m_u, aug_ranks_m_v)) .= -transpose(Umhat)*reshape(v, nxm, nΩm)*Vmhat
        # compute A₀Ŝ₀
        S₀_hat = @view(system.tmp.tmp1[1:aug_ranks_p_u*aug_ranks_p_v + aug_ranks_m_u*aug_ranks_m_v])
        Sp₀_hat = @view(S₀_hat[1:aug_ranks_p_u*aug_ranks_p_v])
        Sm₀_hat = @view(S₀_hat[aug_ranks_p_u*aug_ranks_p_v+1:end])
        reshape(Sp₀_hat, aug_ranks_p_u, aug_ranks_p_v) .= Mp * Sp₀ * transpose(Np)
        reshape(Sm₀_hat, aug_ranks_m_u, aug_ranks_m_v) .= Mm * Sm₀ * transpose(Nm)

        if !isnothing(system.conserved_quantities)
            cons_quant = zeros(size(system.conserved_quantities.p.U, 2))
            for i in 1:size(system.conserved_quantities.p.U, 2)
                cons_quant[i] = transpose(system.conserved_quantities.p.U[:, i]) * Uphat * reshape(Sp₀_hat, aug_ranks_p_u, aug_ranks_p_v) * transpose(Vphat) * system.conserved_quantities.p.V[:, i]
            end
            @show "before S", cons_quant
        end
        mul!(rhs_S, system.mats.rhsBMᵤᵥ, S₀_hat, -1, true)
        # @show rhs_S

    end    
    CUDA.NVTX.@range "S-step" begin
        S₁ = @view(system.tmp.tmp1[1:aug_ranks_p_u*aug_ranks_p_v + aug_ranks_m_u*aug_ranks_m_v])

        mul!(S₁, system.mats.inv_matBMᵤᵥ, rhs_S, true, false)
    end
    CUDA.NVTX.@range "truncate" begin
        Sp₁_hat = reshape(@view(S₁[1:aug_ranks_p_u*aug_ranks_p_v]), (aug_ranks_p_u, aug_ranks_p_v))
        Sm₁_hat = reshape(@view(S₁[aug_ranks_p_u*aug_ranks_p_v+1:end]), (aug_ranks_m_u, aug_ranks_m_v))

        # debugging
        # mass before truncation
        if !isnothing(system.conserved_quantities)
            cons_quant = zeros(size(system.conserved_quantities.p.U, 2))
            for i in 1:size(system.conserved_quantities.p.U, 2)
                cons_quant[i] = transpose(system.conserved_quantities.p.U[:, i]) * Uphat * Sp₁_hat * transpose(Vphat) * system.conserved_quantities.p.V[:, i]
            end
            @show "before trunc", cons_quant
        end

        # NEW STRATEGY
        @show "before fix", dot(Sp₁_hat, Sp₁_hat)
        svd_p = svd(Sp₁_hat)
        svd_m = svd(Sm₁_hat)
        
        ranks = _compute_new_ranks(system, svd_p.S, svd_m.S)
        if !isnothing(system.conserved_quantities)
            ranks = (p = min(ranks.p, system.max_ranks.p - size(system.conserved_quantities.p.U, 2)), m = min(ranks.m, system.max_ranks.m - size(system.conserved_quantities.m.U, 2)))
            Pp = _orthonormalize(transpose(Uphat) * system.conserved_quantities.p.U, svd_p.U[:, 1:ranks.p])
            Qp = _orthonormalize(transpose(Vphat) * system.conserved_quantities.p.V, svd_p.V[:, 1:ranks.p])
            Pm = _orthonormalize(transpose(Umhat) * system.conserved_quantities.m.U, svd_m.U[:, 1:ranks.m])
            Qm = _orthonormalize(transpose(Vmhat) * system.conserved_quantities.m.V, svd_m.V[:, 1:ranks.m])
        else
            Pp = svd_p.U[:, 1:ranks.p]
            Qp = svd_p.V[:, 1:ranks.p]
            Pm = svd_m.U[:, 1:ranks.m]
            Qm = svd_m.V[:, 1:ranks.m]
        end

        @assert size(Pp, 2) == size(Qp, 2) # can we relax even this? then the svd would be non square from the beginning.. 
        @assert size(Pm, 2) == size(Qm, 2)
        x.ranks.p[] = size(Pp, 2)
        x.ranks.m[] = size(Pm, 2)
        ((Up₁, Sp₁, Vtp₁), (Um₁, Sm₁, Vtm₁)) = USVt(x)
        @show x.ranks

        mul!(Up₁, Uphat, Pp)
        mul!(Um₁, Umhat, Pm)
        mul!(Vtp₁, transpose(Qp), transpose(Vphat))
        mul!(Vtm₁, transpose(Qm), transpose(Vmhat))

        Sp₁ .= transpose(Pp)*Sp₁_hat*Qp
        Sm₁ .= transpose(Pm)*Sm₁_hat*Qm

        @show "after fix", dot(Sp₁, Sp₁)


        # OLD STRATEGY
        # Pp, Sp, Qp = svd(Sp₁_hat)
        # Pm, Sm, Qm = svd(Sm₁_hat)

        # ranks = _compute_new_ranks(system, Sp, Sm)
        # x.ranks.p[], x.ranks.m[] = ranks.p, ranks.m
        # ((Up₁, Sp₁, Vtp₁), (Um₁, Sm₁, Vtm₁)) = USVt(x)
        # @show ranks

        # mul!(Up₁, Uphat, @view(Pp[1:aug_ranks_p_u, 1:ranks.p]))
        # mul!(Um₁, Umhat, @view(Pm[1:aug_ranks_m_u, 1:ranks.m]))
        # mul!(Vtp₁, @view(adjoint(Qp)[1:ranks.p, 1:aug_ranks_p_v]), transpose(Vphat))
        # mul!(Vtm₁, @view(adjoint(Qm)[1:ranks.m, 1:aug_ranks_m_v]), transpose(Vmhat))
        # copyto!(Sp₁, Diagonal(@view(Sp[1:ranks.p])))
        # copyto!(Sm₁, Diagonal(@view(Sm[1:ranks.m])))

        # # fix conservation
        # if !isnothing(system.conserved_quantities)
        #     @show "before fix", dot(Sp₁_hat, Sp₁_hat), dot(Sp₁, Sp₁)
        #     _preserve_invariant!(Sp₁, Sp₁_hat, @view(Pp[1:aug_ranks_p_u, 1:ranks.p]), @view(adjoint(Qp)[1:ranks.p, 1:aug_ranks_p_v]), transpose(Uphat) * system.conserved_quantities.p.U, transpose(Vphat) * system.conserved_quantities.p.V)
        #     @show "after fix", dot(Sp₁_hat, Sp₁_hat), dot(Sp₁, Sp₁)
        # end
        # if  !isnothing(system.conserved_quantities)
        #     _preserve_invariant!(Sm₁, Sm₁_hat, @view(Pm[1:aug_ranks_m_u, 1:ranks.m]), @view(adjoint(Qm)[1:ranks.m, 1:aug_ranks_m_v]), transpose(Umhat) * system.conserved_quantities.m.U, transpose(Vmhat) * system.conserved_quantities.m.V)
        # end

        # mass after truncation
        if !isnothing(system.conserved_quantities)
            cons_quant = zeros(size(system.conserved_quantities.p.U, 2))
            for i in 1:size(system.conserved_quantities.p.U, 2)
                cons_quant[i] = transpose(system.conserved_quantities.p.U[:, i]) * Up₁ * Sp₁ * Vtp₁ * system.conserved_quantities.p.V[:, i]
            end
            @show "after trunc", cons_quant
        end
    end
end

# inputs A: full matrix; U, S, Vt: truncated SVD; u, v: invariant vectors
function _preserve_invariant!(S, A, U, Vt, u, v)
    @assert size(u, 2) == size(v, 2)
    m = size(u, 2)
    if iszero(m) return S end
    u_tilde = transpose(U)*u
    v_tilde = Vt*v
    δ = zeros(m)
    for i in 1:m
        δ[i] = dot(@view(u[:, i]), A, @view(v[:, i])) - dot(@view(u_tilde[:, i]), S, @view(v_tilde[:, i]))
    end
    A_mat = zeros(m, m*m)
    for i in 1:m
        A_mat[i, :] = kron(transpose(transpose(v_tilde)*v_tilde[:, i]), transpose(u_tilde[:, i])*u_tilde)
    end
    W_vec = A_mat \ δ
    W = reshape(W_vec, m, m)
    S .+= u_tilde * W * transpose(v_tilde)
end

function allocate_solution_vector(system::DiscreteDLRPNSystem5)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    max_ranks = system.max_ranks
    ranks = adaptive(system) ? (p=Ref(2), m=Ref(2)) : (p=Ref(max_ranks.p), m=Ref(max_ranks.m))# TODO: the 2 here is arbitrary (choose from boundary condition rank?)
    return LowwRankSolution(
        (U = allocate_vec(arch, nxp*max_ranks.p), S = allocate_vec(arch, max_ranks.p*max_ranks.p), Vt = allocate_vec(arch, max_ranks.p*nΩp)),
        (U = allocate_vec(arch, nxm*max_ranks.m), S = allocate_vec(arch, max_ranks.m*max_ranks.m), Vt = allocate_vec(arch, max_ranks.m*nΩm)),
        max_ranks, 
        ranks,
        n_basis(system.problem)
    )
end
