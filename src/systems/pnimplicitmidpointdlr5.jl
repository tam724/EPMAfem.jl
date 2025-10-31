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

function init_basis_augmentation(pbl, aug::Symbol)
    if aug == :mass
        nb = EPMAfem.n_basis(pbl)
        basis_augmentation = (p=(V = EPMAfem.allocate_mat(architecture(pbl), nb.nΩ.p, 1), ),
                              m=(V = EPMAfem.allocate_mat(architecture(pbl), nb.nΩ.m, 0), ))
        Ωp, _ = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(pbl.model), Ω->1)
        copy!(@view(basis_augmentation.p.V[:, 1]), Ωp)
        basis_augmentation.p.V .= qr(basis_augmentation.p.V).Q |> mat_type(architecture(pbl))
        return (1, 0), basis_augmentation
    else
        error("$aug ")
    end 
end

function init_basis_augmentation(_, ::Nothing)
    return (0, 0), nothing
end

function init_basis_augmentation(_, basis_augmentation)
    return (size(basis_augmentation.p.V, 2), size(basis_augmentation.m.V, 2)), basis_augmentation
end
        

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

    (n_bas_aug_p, n_bas_aug_m), basis_augmentation = init_basis_augmentation(pbl, basis_augmentation)

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

function _compute_new_rank(S_, tol, r_max)
    S = collect(S_)
    Σσ² = 0.0
    r = length(S)
    norm_S² = sum(S.^2)
    while ((Σσ² + S[r]^2 < tol^2*norm_S²) || (sqrt(Σσ² + S[r]^2) < 1e-14)) && r > 1
        Σσ² += S[r]^2
        r = r - 1
    end
    return min(r_max, r)
end

function _compute_new_ranks(system::DiscreteDLRPNSystem5{<:Real}, Sp, Sm)
    return (
        p = _compute_new_rank(Sp, system.tolerance, system.max_ranks.p),
        m = _compute_new_rank(Sm, system.tolerance, system.max_ranks.m)
    )
end

_compute_new_ranks(system::DiscreteDLRPNSystem5{<:Nothing}, _, _) = system.max_ranks

function _orthonormalize(arch, bases...; tol=1e-15)
    return qr(hcat(bases...)).Q |> mat_type(arch)
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

        Vphat = _orthonormalize(architecture(system.problem), transpose(Ltp₁), transpose(Vtp₀))
        Vmhat = _orthonormalize(architecture(system.problem), transpose(Ltm₁), transpose(Vtm₀))
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

        Uphat = _orthonormalize(architecture(system.problem), Kp₁, Up₀)
        Umhat = _orthonormalize(architecture(system.problem), Km₁, Um₀)
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

        Vphat = _orthonormalize(architecture(system.problem), transpose(Vtp₀), transpose(Ltp₁))
        Vmhat = _orthonormalize(architecture(system.problem), transpose(Vtm₀), transpose(Ltm₁))
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
            # @show sum(Kp₀ * transpose(Vphat) * system.basis_augmentation.p.V)
        end
    end
    CUDA.NVTX.@range "aug K-step" begin
        # solve the K-step linear system
        K₁, Kp₁, Km₁ = _decompose_pm(system.tmp.tmp1, ((nxp, aug_ranks_v.p), (nxm, aug_ranks_v.m)))
        mul!(K₁, system.mats.inv_matBMᵥ, rhs_K, true, false)
    end
    # @show sum(Kp₁ * transpose(Vphat) * system.basis_augmentation.p.V)

    CUDA.NVTX.@range "truncate" begin
        Wphat = transpose(Vphat) * system.basis_augmentation.p.V
        Wmhat = transpose(Vmhat) * system.basis_augmentation.m.V

        VRptilde = qr(Vphat .- system.basis_augmentation.p.V * transpose(Wphat))
        VRmtilde = qr(Vmhat .- system.basis_augmentation.m.V * transpose(Wmhat))

        Pp, Sp, Qp = svd(Kp₁*transpose(VRptilde.R))
        Pm, Sm, Qm = svd(Km₁*transpose(VRmtilde.R))

        ranks = _compute_new_ranks(system, Sp, Sm)
        n_aug_p, n_aug_m = size(system.basis_augmentation.p.V, 2), size(system.basis_augmentation.m.V, 2)
        ranks = (p=min(system.max_ranks.p, ranks.p+n_aug_p), m=min(ranks.m, ranks.m+n_aug_m))
        x.ranks.p[], x.ranks.m[] = ranks.p, ranks.m
        ((Up₁, Sp₁, Vtp₁), (Um₁, Sm₁, Vtm₁)) = USVt(x)

        URp = qr([Kp₁*Wphat Pp[:, 1:ranks.p-n_aug_p]])
        URm = qr([Km₁*Wmhat Pm[:, 1:ranks.m-n_aug_m]])

        m_type = mat_type(architecture(system.problem))
        Up₁ .= m_type(URp.Q)
        Um₁ .= m_type(URm.Q)

        Vtp₁ .= transpose([system.basis_augmentation.p.V m_type(VRptilde.Q)*Qp[:, 1:ranks.p-n_aug_p]])
        Vtm₁ .= transpose([system.basis_augmentation.m.V m_type(VRmtilde.Q)*Qm[:, 1:ranks.m-n_aug_m]])

        @view(Sp₁[:, 1:n_aug_p]) .= @view(URp.R[:, 1:n_aug_p])
        @view(Sm₁[:, 1:n_aug_m]) .= @view(URm.R[:, 1:n_aug_m])
        
        mul!(@view(Sp₁[:, n_aug_p+1:end]), @view(URp.R[:, n_aug_p+1:end]), Diagonal(Sp[1:ranks.p-n_aug_p]))
        mul!(@view(Sm₁[:, n_aug_m+1:end]), @view(URm.R[:, n_aug_m+1:end]), Diagonal(Sm[1:ranks.m-n_aug_m]))
    end
    # @show sum(Up₁ * Sp₁ * Vtp₁ * system.basis_augmentation.p.V)
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
