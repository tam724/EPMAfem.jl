Base.@kwdef @concrete struct DiscreteDLRPNSystem5{ADAPTIVE, AUGMENT} <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp

    max_ranks
    tolerance::ADAPTIVE
    basis_augmentation::AUGMENT
end

function Base.adjoint(A::DiscreteDLRPNSystem5{ADAPTIVE, AUGMENT}) where {ADAPTIVE, AUGMENT}
    return DiscreteDLRPNSystem5(adjoint=!A.adjoint, problem=A.problem, coeffs=A.coeffs, mats=A.mats, rhs=A.rhs, tmp=A.tmp, max_ranks=A.max_ranks, tolerance=A.tolerance, basis_augmentation=A.basis_augmentation)
end

adaptive(::DiscreteDLRPNSystem5{Nothing}) = false
adaptive(::DiscreteDLRPNSystem5{<:Real}) = true

augmented(::DiscreteDLRPNSystem5{<:Any, Nothing}) = false
augmented(::DiscreteDLRPNSystem5{<:Any, AUGMENT}) where AUGMENT = true

using EPMAfem.PNLazyMatrices: only_unique

function implicit_midpoint_dlr5(pbl::DiscretePNProblem; solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=nothing, basis_augmentation=nothing)
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

    n_bas_aug_p = isnothing(basis_augmentation) ? 0 : max(size(basis_augmentation.p.U, 2), size(basis_augmentation.p.V, 2))
    n_bas_aug_m = isnothing(basis_augmentation) ? 0 : max(size(basis_augmentation.m.U, 2), size(basis_augmentation.m.V, 2))

    max_aug_ranks = (p=3max_ranks.p+n_bas_aug_p, m=3max_ranks.m+n_bas_aug_m)

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

    # inv_matBM = solver([A(cfs.mat) B
    #     transpose(B) C(cfs.mat)])

    # rhsBM = [A(cfs.rhs) B
    #     transpose(B) C(cfs.rhs)]

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

        # inv_matBM = inv_matBM,
        # rhsBM = rhsBM,

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
        diagR = abs.(diag(R)) # TODO: make this a cutoff again ? 
        Qi_clean = Matrix(Qi_orth)#[:, diagR .> tol]
        
        Q_combined = hcat(Q_combined, Qi_clean)
    end

    return Matrix(qr(Q_combined).Q)
end

function tsvd(A, r)
    svd_ = svd(A)
    return LinearAlgebra.SVD(svd_.U[:, 1:r], svd_.S[1:r], svd_.Vt[1:r, :])
end

# computes the rank-(length(M)) update ΔS on S such that diag(transpose(Mᵤ) * U (S + ΔS) * transpose(V) * Mᵥ) ≈ M
function _rank_M_update(S, U, V, Mᵤ, Mᵥ, M)
    A = transpose(Mᵤ) * U      # r × u
    B = transpose(V) * Mᵥ      # v × r

    # Current approximation
    M̂ = diag(A * S * B)
    R = M - M̂

    r, u = size(A)
    v, r2 = size(B)
    @assert r == r2

    C = zeros(r, u * v)

    # Build C matrix row-by-row
    for i in 1:r
        C[i, :] = kron(B[:, i]', A[i, :])  # Row i is A[i,:] ⊗ B[:,i]'
    end

    # Solve least squares: C * vec(ΔS) = R
    vec_ΔS = C \ R
    ΔS = reshape(vec_ΔS, u, v)

    return ΔS
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

function _decompose_pm(v::AbstractVector, ((mp, np), (mm, nm))::NTuple{2, Tuple{<:Integer, <:Integer}})
    X = @view(v[1:mp*np + mm*nm])
    Xp = reshape(@view(X[1:mp*np]), (mp, np))
    Xm = reshape(@view(X[mp*np+1:end]), (mm, nm))
    return X, Xp, Xm
end

# the robust rank-adaptive BUG INTEGRATOR
function _step!(x, system::DiscreteDLRPNSystem5{<:Any, Nothing}, rhs_ass::PNVectorAssembler, idx, Δϵ)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)

    CUDA.NVTX.@range "rhs assembly" begin
        assemble_at!(system.rhs.full, rhs_ass, idx, Δϵ, true)
        u = @view(system.rhs.full[1:nxp*nΩp])
        v = @view(system.rhs.full[nxp*nΩp+1:end])
    end

    ((Up₀, Sp₀, Vtp₀), (Um₀, Sm₀, Vtm₀)) = USVt(x)
    ranks = (p=x.ranks.p[], m=x.ranks.m[])


    CUDA.NVTX.@range "L-step prep" begin
        PNLazyMatrices.set!(system.mats.Up, vec(Up₀), size(Up₀))
        PNLazyMatrices.set!(system.mats.Um, vec(Um₀), size(Um₀))

        rhs_Lt, rhs_Ltp, rhs_Ltm = _decompose_pm(system.rhs.proj, ((ranks.p, nΩp), (ranks.m, nΩm)))

        # compute û (the projected source/bc)
        mul!(rhs_Ltp, transpose(Up₀), reshape(u, nxp, nΩp), -1, false)
        mul!(rhs_Ltm, transpose(Um₀), reshape(v, nxm, nΩm), -1, false)

        let (Lt₀, Ltp₀, Ltm₀) = _decompose_pm(system.tmp.tmp1, ((ranks.p, nΩp), (ranks.m, nΩm)))
            # compute L0
            mul!(Ltp₀, Sp₀, Vtp₀)
            mul!(Ltm₀, Sm₀, Vtm₀)

            # update the rhs from the previous energy/time step
            mul!(rhs_Lt, system.mats.rhsBMᵤ, Lt₀, -1, true)
        end
    end
    CUDA.NVTX.@range "L-step" begin
        # solve the L-step linear system
        Lt₁, Ltp₁, Ltm₁ = _decompose_pm(system.tmp.tmp1, ((ranks.p, nΩp), (ranks.m, nΩm)))
        mul!(Lt₁, system.mats.inv_matBMᵤ, rhs_Lt, true, false)

        Vphat = _orthonormalize(transpose(Ltp₁), transpose(Vtp₀))
        Vmhat = _orthonormalize(transpose(Ltm₁), transpose(Vtm₀))
        Np = transpose(Vphat)*transpose(Vtp₀)
        Nm = transpose(Vmhat)*transpose(Vtm₀)
    end

    CUDA.NVTX.@range "K-step prep" begin
        PNLazyMatrices.set!(system.mats.Vtp, vec(Vtp₀), size(Vtp₀))
        PNLazyMatrices.set!(system.mats.Vtm, vec(Vtm₀), size(Vtm₀))

        rhs_K, rhs_Kp, rhs_Km = _decompose_pm(system.rhs.proj, ((nxp, ranks.p), (nxm, ranks.m)))

        # compute û (the projected source/bc)
        mul!(rhs_Kp, reshape(u, nxp, nΩp), transpose(Vtp₀), -1, false)
        mul!(rhs_Km, reshape(v, nxm, nΩm), transpose(Vtm₀), -1, false)

        let (K₀, Kp₀, Km₀) = _decompose_pm(system.tmp.tmp1, ((nxp, ranks.p), (nxm, ranks.m)))
            # compute K0
            mul!(Kp₀, Up₀, Sp₀)
            mul!(Km₀, Um₀, Sm₀)

            #update the rhs from the previous energy/time step
            mul!(rhs_K, system.mats.rhsBMᵥ, K₀, -1, true)
        end
    end
    CUDA.NVTX.@range "K-step" begin
        # solve the K-step linear system
        K₁, Kp₁, Km₁ = _decompose_pm(system.tmp.tmp1, ((nxp, ranks.p), (nxm, ranks.m)))
        mul!(K₁, system.mats.inv_matBMᵥ, rhs_K, true, false)

        Uphat = _orthonormalize(Kp₁, Up₀)
        Umhat = _orthonormalize(Km₁, Um₀)
        Mp = transpose(Uphat)*Up₀
        Mm = transpose(Umhat)*Um₀
    end

    CUDA.NVTX.@range "S-step prep" begin
        Vtphat_ = transpose(Vphat) |> mat_type(architecture(system.problem)) # todo
        Vtmhat_ = transpose(Vmhat) |> mat_type(architecture(system.problem))
        PNLazyMatrices.set!(system.mats.Vtp, vec(Vtphat_), size(Vtphat_))
        PNLazyMatrices.set!(system.mats.Vtm, vec(Vtmhat_), size(Vtmhat_))
        PNLazyMatrices.set!(system.mats.Up, vec(Uphat), size(Uphat))
        PNLazyMatrices.set!(system.mats.Um, vec(Umhat), size(Umhat))

        aug_ranks = ((size(Uphat, 2), size(Vphat, 2)), (size(Umhat, 2), size(Vmhat, 2)))
        rhs_S, rhs_Sp, rhs_Sm = _decompose_pm(system.rhs.proj, aug_ranks)

        # compute û (the projected source/bc)
        rhs_Sp .= -transpose(Uphat)*reshape(u, nxp, nΩp)*Vphat
        rhs_Sm .= -transpose(Umhat)*reshape(v, nxm, nΩm)*Vmhat

        let (Shat₀, Sphat₀, Smhat₀) = _decompose_pm(system.tmp.tmp1, aug_ranks)
            # compute Shat0
            Sphat₀ .= Mp * Sp₀ * transpose(Np)
            Smhat₀ .= Mm * Sm₀ * transpose(Nm)

            #update the rhs from the previous energy/time step
            mul!(rhs_S, system.mats.rhsBMᵤᵥ, Shat₀, -1, true)
        end
    end    
    CUDA.NVTX.@range "S-step" begin
        # solve the S-step linear system
        Shat₁, Sphat₁, Smhat₁ = _decompose_pm(system.tmp.tmp1, aug_ranks)
        mul!(Shat₁, system.mats.inv_matBMᵤᵥ, rhs_S, true, false)
    end
    CUDA.NVTX.@range "truncate" begin
        Pp, Sp, Qp = svd(Sphat₁)
        Pm, Sm, Qm = svd(Smhat₁)
        
        ranks = _compute_new_ranks(system, Sp, Sm)
        x.ranks.p[], x.ranks.m[] = ranks.p, ranks.m
        ((Up₁, Sp₁, Vtp₁), (Um₁, Sm₁, Vtm₁)) = USVt(x)
        @show ranks

        mul!(Up₁, Uphat, @view(Pp[:, 1:ranks.p]))
        mul!(Um₁, Umhat, @view(Pm[:, 1:ranks.m]))
        mul!(Vtp₁, @view(adjoint(Qp)[1:ranks.p, :]), transpose(Vphat))
        mul!(Vtm₁, @view(adjoint(Qm)[1:ranks.m, :]), transpose(Vmhat))
        copyto!(Sp₁, Diagonal(@view(Sp[1:ranks.p])))
        copyto!(Sm₁, Diagonal(@view(Sm[1:ranks.m])))
    end
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

    CUDA.NVTX.@range "L-step prep" begin
        PNLazyMatrices.set!(system.mats.Up, vec(Up₀), size(Up₀))
        PNLazyMatrices.set!(system.mats.Um, vec(Um₀), size(Um₀))

        rhs_Lt, rhs_Ltp, rhs_Ltm = _decompose_pm(system.rhs.proj, ((ranks.p, nΩp), (ranks.m, nΩm)))

        # compute û (the projected source/bc)
        mul!(rhs_Ltp, transpose(Up₀), reshape(u, nxp, nΩp), -1, false)
        mul!(rhs_Ltm, transpose(Um₀), reshape(v, nxm, nΩm), -1, false)

        let (Lt₀, Ltp₀, Ltm₀) = _decompose_pm(system.tmp.tmp1, ((ranks.p, nΩp), (ranks.m, nΩm)))
            # compute L0
            mul!(Ltp₀, Sp₀, Vtp₀)
            mul!(Ltm₀, Sm₀, Vtm₀)

            # update the rhs from the previous energy/time step
            mul!(rhs_Lt, system.mats.rhsBMᵤ, Lt₀, -1, true)
        end
    end
    CUDA.NVTX.@range "L-step" begin
        # solve the L-step linear system
        Lt₁, Ltp₁, Ltm₁ = _decompose_pm(system.tmp.tmp1, ((ranks.p, nΩp), (ranks.m, nΩm)))
        mul!(Lt₁, system.mats.inv_matBMᵤ, rhs_Lt, true, false)

        Vphat = _orthonormalize(transpose(Ltp₁), transpose(Vtp₀))
        Vmhat = _orthonormalize(transpose(Ltm₁), transpose(Vtm₀))
        Np = transpose(Vphat)*transpose(Vtp₀)
        Nm = transpose(Vmhat)*transpose(Vtm₀)
        aug_ranks_v = (p=size(Vphat, 2), m=size(Vmhat, 2))
    end

    CUDA.NVTX.@range "aug K-step prep" begin
        PNLazyMatrices.set!(system.mats.Vtp, vec(Matrix(transpose(Vphat))), size(transpose(Vphat))) # TODO: should not copy
        PNLazyMatrices.set!(system.mats.Vtm, vec(Matrix(transpose(Vmhat))), size(transpose(Vmhat)))

        rhs_K, rhs_Kp, rhs_Km = _decompose_pm(system.rhs.proj, ((nxp, aug_ranks_v.p), (nxm, aug_ranks_v.m)))

        # compute û (the projected source/bc)
        mul!(rhs_Kp, reshape(u, nxp, nΩp), Vphat, -1, false)
        mul!(rhs_Km, reshape(v, nxm, nΩm), Vmhat, -1, false)

        let (K₀, Kp₀, Km₀) = _decompose_pm(system.tmp.tmp1, ((nxp, aug_ranks_v.p), (nxm, aug_ranks_v.m)))
            # compute K0
            mul!(Kp₀, Up₀ * Sp₀, transpose(Np)) # TODO: temporary storage for U*S ? 
            mul!(Km₀, Um₀ * Sm₀, transpose(Nm))

            #update the rhs from the previous energy/time step
            mul!(rhs_K, system.mats.rhsBMᵥ, K₀, -1, true)
        end
    end
    CUDA.NVTX.@range "aug K-step" begin
        # solve the K-step linear system
        K₁, Kp₁, Km₁ = _decompose_pm(system.tmp.tmp1, ((nxp, aug_ranks_v.p), (nxm, aug_ranks_v.m)))
        mul!(K₁, system.mats.inv_matBMᵥ, rhs_K, true, false)
    end

    CUDA.NVTX.@range "truncate" begin
        vphat = transpose(Vphat) * system.basis_augmentation.p.V
        vmhat = transpose(Vmhat) * system.basis_augmentation.m.V
        Up_cons = qr(Kp₁ * vphat)
        Um_cons = qr(Km₁ * vmhat)
        Kp₁_tilde = qr(Kp₁ .- Kp₁*vphat*(vphat')) # project out the mass carrying mode(s)
        Km₁_tilde = qr(Km₁ .- Km₁*vmhat*(vmhat'))

        Pp, Sp, Qp = svd(Kp₁_tilde.R)
        Pm, Sm, Qm = svd(Km₁_tilde.R)

        ranks = _compute_new_ranks(system, Sp, Sm)
        n_aug_p = size(system.basis_augmentation.p.V, 2)
        n_aug_m = size(system.basis_augmentation.m.V, 2)
        ranks = (p=min(system.max_ranks.p, ranks.p+n_aug_p), m=min(ranks.m, ranks.m+n_aug_m))
        x.ranks.p[], x.ranks.m[] = ranks.p, ranks.m
        ((Up₁, Sp₁, Vtp₁), (Um₁, Sm₁, Vtm₁)) = USVt(x)

        σp = Up_cons.R;
        Up₁_ = qr([Matrix(Up_cons.Q) Kp₁_tilde.Q*Pp[:, 1:ranks.p-n_aug_p]])
        Vp₁_ = qr([system.basis_augmentation.p.V Vphat*Qp[:, 1:ranks.p-n_aug_p]])
        Sp₁_ = [
            σp zeros(n_aug_p, ranks.p-n_aug_p)
            zeros(ranks.p-n_aug_p, n_aug_p) Diagonal(Sp[1:ranks.p-n_aug_p])
        ]

        Up₁ .= Matrix(Up₁_.Q)
        Vtp₁ .= transpose(Matrix(Vp₁_.Q))
        Sp₁ .= Matrix(Up₁_.R) * Sp₁_ * transpose(Matrix(Vp₁_.R))
        
        # same for m
        σm = Um_cons.R;
        Um₁_ = qr([Matrix(Um_cons.Q) Km₁_tilde.Q*Pm[:, 1:ranks.m-n_aug_m]])
        Vm₁_ = qr([system.basis_augmentation.m.V Vmhat*Qm[:, 1:ranks.m-n_aug_m]])
        Sm₁_ = [
            σm zeros(n_aug_m, ranks.m-n_aug_m)
            zeros(ranks.m-n_aug_m, n_aug_m) Diagonal(Sm[1:ranks.m-n_aug_m])
        ]

        Um₁ .= Matrix(Um₁_.Q)
        Vtm₁ .= transpose(Matrix(Vm₁_.Q))
        Sm₁ .= Matrix(Um₁_.R) * Sm₁_ * transpose(Matrix(Vm₁_.R))
    end
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
