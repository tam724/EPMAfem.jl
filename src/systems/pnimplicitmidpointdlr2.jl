Base.@kwdef @concrete struct DiscreteDLRPNSystem2 <: AbstractDiscretePNSystem
    adjoint::Bool=false
    problem

    coeffs
    mats
    rhs
    tmp
    tmp2

    max_rank
end

function implicit_midpoint_dlr2(pbl::DiscretePNProblem; max_rank=20)
    arch = architecture(pbl)

    T = base_type(architecture(pbl))
    ns = EPMAfem.n_sums(pbl)
    nb = EPMAfem.n_basis(pbl)
    Δϵ = step(energy_model(pbl.model))

    Vt = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, max_rank, nb.nΩ.p), (Ref(max_rank), Ref(nb.nΩ.p)))
    V = transpose(Vt)
    U = PNLazyMatrices.LazyResizeMatrix(allocate_mat(arch, nb.nx.p, max_rank), (Ref(nb.nx.p), Ref(max_rank)))
    Ut = transpose(U)

    coeffs = (a = [LazyScalar(zero(T)) for _ in 1:ns.ne], c = [[LazyScalar(zero(T)) for _ in 1:ns.nσ] for _ in 1:ns.ne], Δ=LazyScalar(T(Δϵ)), γ=LazyScalar(T(0.5)), δ=LazyScalar(T(-0.5)), δt=LazyScalar(T(0.5)))
    ρp, ρm, ∂p, ∇pm = lazy_space_matrices(pbl)
    Ip, Im, kp, km, absΩp, Ωpm = lazy_direction_matrices(pbl)

    A_Ikp(i) = cache(coeffs.a[i]*Ip + sum(coeffs.c[i][j]*kp[i][j] for j in 1:ns.nσ))
    C_Ikm(i) = cache(coeffs.a[i]*Im + sum(coeffs.c[i][j]*km[i][j] for j in 1:ns.nσ))
    C = sum(kron_AXB(ρm[i], C_Ikm(i)) for i in 1:ns.ne)

    A_V = sum(kron_AXB(ρp[i], cache(Vt*A_Ikp(i)*V)) for i in 1:ns.ne)
    B_V = sum(kron_AXB(∇pm[i], cache(Ωpm[i]*V)) for i in 1:ns.nd)
    D_V = sum(kron_AXB(∂p[i], cache(Vt*absΩp[i]*V)) for i in 1:ns.nd)

    BM_V = [
        coeffs.Δ*(A_V + coeffs.γ*D_V) coeffs.Δ*(coeffs.δ*B_V)
        -(coeffs.Δ*(coeffs.δt*transpose(B_V))) -(coeffs.Δ*C)
    ]
    half_BM_V⁻¹ = PNLazyMatrices.half_schur_complement(BM_V, Krylov.minres, cache ∘ LinearAlgebra.inv!)

    A_U = sum(kron_AXB(cache(Ut*ρp[i]*U), A_Ikp(i)) for i in 1:ns.ne)
    B_U = sum(kron_AXB(cache(Ut*∇pm[i]), Ωpm[i]) for i in 1:ns.nd)
    D_U = sum(kron_AXB(cache(Ut*∂p[i]*U), absΩp[i]) for i in 1:ns.nd)

    BM_U = [
        coeffs.Δ*(A_U + coeffs.γ*D_U) coeffs.Δ*(coeffs.δ*B_U)
        -(coeffs.Δ*(coeffs.δt*transpose(B_U))) -(coeffs.Δ*C)
    ]
    half_BM_U⁻¹ = PNLazyMatrices.half_schur_complement(BM_U, Krylov.minres, cache ∘ LinearAlgebra.inv!)

    A_UV = sum(kron_AXB(cache(Ut*ρp[i]*U), cache(Vt*A_Ikp(i)*V)) for i in 1:ns.ne)
    B_UV = sum(kron_AXB(cache(Ut*∇pm[i]), cache(Ωpm[i]*V)) for i in 1:ns.nd)
    D_UV = sum(kron_AXB(cache(Ut*∂p[i]*U), cache(Vt*absΩp[i]*V)) for i in 1:ns.nd)

    BM_UV = [
        coeffs.Δ*(A_UV + coeffs.γ*D_UV) coeffs.Δ*(coeffs.δ*B_UV)
        -(coeffs.Δ*(coeffs.δt*transpose(B_UV))) -(coeffs.Δ*C)
    ]
    BM_UV⁻¹ = PNLazyMatrices.schur_complement(BM_UV, Krylov.minres, cache ∘ LinearAlgebra.inv!)

    uBM_U, uhalf_BM_U⁻¹, uBM_V, uhalf_BM_V⁻¹, uBM_UV, uBM_UV⁻¹, coeffs_, Vt_, U_ = unlazy((BM_U, half_BM_U⁻¹, BM_V, half_BM_V⁻¹, BM_UV, BM_UV⁻¹, coeffs, Vt, U), vec_size -> allocate_vec(arch, vec_size))
    rhs = allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
    tmp = allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
    tmp2 = allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)

    return DiscreteDLRPNSystem2(
        problem = pbl,
        coeffs = coeffs_,
        mats = (BM_U=uBM_U, half_BM_U⁻¹=uhalf_BM_U⁻¹, BM_V=uBM_V, half_BM_V⁻¹=uhalf_BM_V⁻¹, BM_UV=uBM_UV, BM_UV⁻¹=uBM_UV⁻¹, Vt=Vt_, U=U_),
        rhs = rhs,
        tmp = tmp,
        tmp2 = tmp2,
        max_rank=max_rank
    )
end

function step_nonadjoint!(x, system::DiscreteDLRPNSystem2, rhs_ass::PNVectorAssembler, idx, Δϵ)
    @show idx
    CUDA.NVTX.@range "prep" begin
        if system.adjoint @warn "Trying to step_nonadjoint with system marked as adjoint" end
        if system.adjoint != _is_adjoint_vector(rhs_ass) @warn "System {$(system.adjoint)} is marked as not compatible with the vector {$(_is_adjoint_vector(rhs_ass))}" end
        # update the rhs (we multiply the whole linear system with Δϵ -> "normalization")
        # implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
        # minus because we have to bring b to the right side of the equation
        CUDA.NVTX.@range "assemble vec" begin
            assemble_at!(system.rhs, rhs_ass, minus½(idx), -Δϵ, true)
        end

        # CUDA.NVTX.@range "assemble rhs" begin
        #     mul!(system.rhs, system.mats.BM, vec!(system.tmp, x), -1.0, true)
        # end
        U₀, S₀, Vt₀ = USVt(x)


        CUDA.NVTX.@range "copy basis" begin
            # implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
            # K-step (prep)
            PNLazyMatrices.resize_copyto!(system.mats.Vt, Vt₀)
            # L-step (prep)
            PNLazyMatrices.resize_copyto!(system.mats.U, U₀)
            (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
            rhsp, rhsm = @view(system.rhs[1:nxp*nΩp]), @view(system.rhs[nxp*nΩp+1:end])
        end
    end

    CUDA.NVTX.@range "k-step" begin
        # K-step
        # build the rhs
        rhs_K = @view(system.tmp[1:nxp*x.rank[] + nxm*nΩm])
        mul!(reshape(@view(rhs_K[1:nxp*x.rank[]]), (nxp, x.rank[])), reshape(rhsp, (nxp, nΩp)), transpose(Vt₀))
        copyto!(@view(rhs_K[nxp*x.rank[]+1:end]), rhsm)

        # bring sol to rhs
        implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
        K₀ = @view(system.tmp2[1:nxp*x.rank[] + nxm*nΩm])
        mul!(rhs_K, system.mats.BM_V, vec_K!(K₀, U₀, S₀, x._xm), -1.0, true)

        # solve K
        implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
        K₁ = @view(system.tmp2[1:nxp*x.rank[]])
        mul!(K₁, system.mats.half_BM_V⁻¹, rhs_K)
        U₁ = (qr(reshape(K₁, (nxp, x.rank[]))).Q |> mat_type(architecture(system.problem)))[1:size(U₀, 1), 1:size(U₀, 2)]
        M = transpose(U₁)*U₀
    end

    CUDA.NVTX.@range "l-step" begin
        # L-step
        # build the rhs
        rhs_Lt = @view(system.tmp[1:x.rank[]*nΩp + nxm*nΩm])
        mul!(reshape(@view(rhs_Lt[1:x.rank[]*nΩp]), (x.rank[], nΩp)), transpose(U₀), reshape(rhsp, (nxp, nΩp)))
        copyto!(@view(rhs_Lt[x.rank[]*nΩp+1:end]), rhsm)

        # bring sol to rhs
        implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
        Lt₀ = @view(system.tmp2[1:x.rank[]*nΩp + nxm*nΩm])
        mul!(rhs_Lt, system.mats.BM_U, vec_L!(Lt₀, S₀, Vt₀, x._xm), -1.0, true)

        # solve L
        implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
        Lt₁ = @view(system.tmp2[1:x.rank[]*nΩp])
        mul!(Lt₁, system.mats.half_BM_U⁻¹, rhs_Lt)
        V₁ = (qr(transpose(reshape(Lt₁, (x.rank[], nΩp)))).Q |> mat_type(architecture(system.problem)))[1:size(Vt₀, 2), 1:size(Vt₀, 1)]
        N = transpose(V₁)*transpose(Vt₀)
    end

    CUDA.NVTX.@range "s-step" begin
        # S-step (prep)
        PNLazyMatrices.resize_copyto!(system.mats.Vt, transpose(V₁))
        PNLazyMatrices.resize_copyto!(system.mats.U, U₁)

        # S-step
        # build the rhs
        rhs_S = @view(system.tmp[1:x.rank[]*x.rank[] + nxm*nΩm])
        copyto!(reshape(@view(rhs_S[1:x.rank[]*x.rank[]]), (x.rank[], x.rank[])), transpose(U₁)*reshape(rhsp, (nxp, nΩp))*V₁)
        copyto!(@view(rhs_S[x.rank[]*x.rank[]+1:end]), rhsm)

        # bring sol to rhs
        implicit_midpoint_coeffs_nonadjoint_rhs!(system.coeffs, system.problem, idx, Δϵ)
        S₀_hat = @view(system.tmp2[1:x.rank[]*x.rank[] + nxm*nΩm])
        mul!(rhs_S, system.mats.BM_UV, vec!(S₀_hat, M, S₀, transpose(N), x._xm), -1.0, true)

        # solve S
        implicit_midpoint_coeffs_nonadjoint_mat!(system.coeffs, system.problem, idx, Δϵ)
        S₁ = mul!(S₀_hat, system.mats.BM_UV⁻¹, rhs_S)
    end

    CUDA.NVTX.@range "finalize" begin
        copyto!(U₀, U₁)
        copyto!(Vt₀, transpose(V₁))
        copyto!(S₀, @view(S₁[1:x.rank[]*x.rank[]]))
        copyto!(x._xm, @view(S₁[x.rank[]*x.rank[]+1:end]))
    end
end

function allocate_solution_vector(system::DiscreteDLRPNSystem2)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(system.problem)
    arch = architecture(system.problem)
    max_rank = system.max_rank
    return LowRankSolution(allocate_vec(arch, nxp*max_rank + max_rank*max_rank + max_rank*nΩp), allocate_vec(arch, nxm*nΩm), max_rank, Ref(max_rank), nxp, nΩp)
end


function vec_K!(x::AbstractVector, Up, Sp, xm)
    @assert size(Sp, 1) == size(Sp, 2)
    nxp = size(Up, 1)
    rank = size(Sp, 2)
    Xp = reshape(@view(x[1:nxp*rank]), nxp, rank)
    mul!(Xp, Up, Sp)
    _xm = @view(x[nxp*rank+1:end])
    _xm .= xm
    return x
end

function vec_L!(x::AbstractVector, Sp, Vtp, xm)
    @assert size(Sp, 1) == size(Sp, 2)
    rank = size(Sp, 1)
    nΩp = size(Vtp, 2)
    Xp = reshape(@view(x[1:rank*nΩp]), rank, nΩp)
    mul!(Xp, Sp, Vtp)
    _xm = @view(x[rank*nΩp+1:end])
    _xm .= xm
    return x
end
