@concrete struct ArrayOfTangentDiscretePNVector <: AbstractDiscretePNVector{true}
    ρp_tangent
    ρm_tangent

    cached_solution
end

function tangent(it::AdjointIterator)
    cached = saveall(it)
    problem = it.system.problem

    ρp_tangent = problem.ρp_tens.skeleton |> problem.arch
    ρm_tangent = problem.ρm_tens.skeleton |> problem.arch
    return ArrayOfTangentDiscretePNVector(ρp_tangent, ρm_tangent, cached)
end

function Base.getindex(arr::ArrayOfTangentDiscretePNVector, i_e, i_x, Δ=true)
    system = arr.cached_solution.it.system
    problem = system.problem
    # T = base_type(architecture(discrete_system.model))
    # VT = vec_type(architecture(discrete_system.model))

    # cv(x) = convert_to_architecture(architecture(discrete_system.model), x)

    onehot = zeros(num_free_dofs(SpaceModels.material(space_model(problem.model))))
    onehot[i_x] = Δ
    Sparse3Tensor.project!(problem.ρp_tens, onehot)
    Sparse3Tensor.project!(problem.ρm_tens, onehot)

    copyto!(nonzeros(arr.ρp_tangent), nonzeros(problem.ρp_tens.skeleton))
    copyto!(nonzeros(arr.ρm_tangent), nonzeros(problem.ρm_tens.skeleton))
    return TangentDiscretePNVector(arr, i_x, i_e)
    # return DiscretePNVector(arr.ρp_tangent[i], arr.ρm_tangent[i], arr.cached_solution)
end

function (arr::ArrayOfTangentDiscretePNVector)(it::NonAdjointIterator)
    system = arr.cached_solution.it.system
    problem = system.problem
    arch = problem.arch

    T = base_type(arch)
    # cv_Int(x) = convert_to_architecture(Int64, architecture(discrete_system.model), x)

    Δϵ = T(step(energy_model(problem.model)))

    isp, jsp = arch.(Int64, Sparse3Tensor.get_ijs(problem.ρp_tens))
    ism, jsm = arch.(Int64, Sparse3Tensor.get_ijs(problem.ρm_tens))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)

    Λtemp = allocate_vec(arch, max(nxp*nΩp, nxm*nΩm))
    Λtempp = reshape(@view(Λtemp[1:nxp*nΩp]), (nxp, nΩp))
    Λtempm = reshape(@view(Λtemp[1:nxm*nΩm]), (nxm, nΩm))

    σtemp = allocate_vec(arch, max(nΩp, nΩm))
    σtempp = Diagonal(@view(σtemp[1:nΩp]))
    σtempm = Diagonal(@view(σtemp[1:nΩm]))

    ΛpΦp = [allocate_vec(arch, length(isp)) for _ in 1:length(problem.ρp)]
    ΛmΦm = [allocate_vec(arch, length(ism)) for _ in 1:length(problem.ρp)]

    skip_initial = true
    write_initial = false
    for (ϵ, i_ϵ) in it
        if skip_initial
            skip_initial = false
            continue
        end
        Φp = pview(current_solution(it.system), problem.model)
        Φm = mview(current_solution(it.system), problem.model)

        Λ_im2p = pview(arr.cached_solution[i_ϵ], problem.model)
        Λ_im2m = mview(arr.cached_solution[i_ϵ], problem.model)
        Λ_ip2p = pview(arr.cached_solution[i_ϵ+1], problem.model)
        Λ_ip2m = mview(arr.cached_solution[i_ϵ+1], problem.model)

        for i_e in 1:1:length(problem.ρp)
            s_i = problem.s[i_e, i_ϵ]
            τ_i = problem.τ[i_e, i_ϵ]

            rmul!(σtempp.diag, false)
            for i in 1:size(problem.σ, 2)
                σtempp.diag .+= problem.σ[i_e, i, i_ϵ] .* problem.kp[i_e][i].diag
            end

            mul!(Λtempp, Λ_ip2p, σtempp, -T(0.5), false)
            mul!(Λtempp, Λ_im2p, σtempp, -T(0.5), true)
            Λtempp .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2p .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2p

            Sparse3Tensor.special_matmul!(ΛpΦp[i_e], isp, jsp, Λtempp, Φp, Δϵ, write_initial)

            rmul!(σtempm.diag, false)
            for i in 1:size(problem.σ, 2)
                σtempm.diag .+= problem.σ[i_e, i, i_ϵ] .* problem.km[i_e][i].diag
            end

            mul!(Λtempm, Λ_ip2m, σtempm, -T(0.5), false)
            mul!(Λtempm, Λ_im2m, σtempm, -T(0.5), true)
            Λtempm .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2m .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2m

            Sparse3Tensor.special_matmul!(ΛmΦm[i_e], ism, jsm, Λtempm, Φm, Δϵ, write_initial)
        end
        write_initial = true
    end
    ρs_adjoint = [zeros(num_free_dofs(SpaceModels.material(space_model(problem.model)))) for _ in 1:length(problem.ρp)]
    for i_e in 1:length(problem.ρp)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], problem.ρp_tens, ΛpΦp[i_e] |> collect, true, true)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], problem.ρm_tens, ΛmΦm[i_e] |> collect, true, true)
    end
    return ρs_adjoint
end

@concrete struct TangentDiscretePNVector <: AbstractDiscretePNVector{true}
    parent
    i_x
    i_e
end

function assemble_rhs!(b, rhs::TangentDiscretePNVector, i, Δ, sym)
    system = rhs.parent.cached_solution.it.system
    problem = system.problem
    arch = system.problem.arch

    T = base_type(arch)
    Δϵ = T(step(energy_model(problem.model)))

    fill!(b, zero(eltype(b)))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    np = nxp*nΩp
    nm = nxm*nΩm

    bp = @view(b[1:np])
    bm = @view(b[np+1:np+nm])

    si = problem.s[rhs.i_e, i]
    τi = problem.τ[rhs.i_e, i]
    σi = problem.σ[rhs.i_e, :, i]

    Λ_im2 = rhs.parent.cached_solution[i]
    Λ_ip2 = rhs.parent.cached_solution[i+1]

    #TODO: move temporary allocations somwhere else
    tmp = allocate_vec(arch, max(np, nm))
    tmp2 = allocate_vec(arch, max(nΩp, nΩm))

    a = [(si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    γ = sym ? -1 : 1

    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], problem.Ip, [problem.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_ip2[1:np]), Δ, false)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], problem.Im, [problem.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_ip2[np+1:np+nm]), γ*Δ, false)

    a = [(-si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], problem.Ip, [problem.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_im2[1:np]), Δ, true)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], problem.Im, [problem.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_im2[np+1:np+nm]), γ*Δ, true)
end
