@concrete terse struct TangentDiscretePNVector <: AbstractDiscretePNVector
    adjoint::Bool
    updatable_problem
    cached_solution
    element_index
    cell_index
end

function tangent(upd_problem::UpdatableDiscretePNProblem, ψ::AbstractDiscretePNSolution)
    # this basically creates the "PNVectors" \dot{a}(\cdot{}, ψ) or \dot{a}(ψ, \cdot{})
    # if ψ is an adjoint solution, this is an nonadjoint vector, is ψ is a nonadjoint solution, this is an adjoint vector 
    (n_e, n_cells) = n_parameters(upd_problem)
    return [TangentDiscretePNVector(_is_adjoint_solution(ψ), upd_problem, ψ, i, j) for i in 1:n_e, j in 1:n_cells]
end

_is_adjoint_vector(b::TangentDiscretePNVector) = b.adjoint

function initialize_integration(b::Array{<:TangentDiscretePNVector})
    upd_problem = first(b).updatable_problem
    problem = upd_problem.problem
    arch = problem.arch
    T = base_type(arch)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)

    isp, jsp = arch.(Int64, Sparse3Tensor.get_ijs(upd_problem.ρp_tens))
    ism, jsm = arch.(Int64, Sparse3Tensor.get_ijs(upd_problem.ρm_tens))
    Λtemp = allocate_vec(arch, max(nxp*nΩp, nxm*nΩm))
    σtemp = allocate_vec(arch, max(nΩp, nΩm))
    ΛpΦp = [allocate_vec(arch, length(isp)) for _ in 1:length(problem.ρp)]
    ΛmΦm = [allocate_vec(arch, length(ism)) for _ in 1:length(problem.ρp)]
    for ΛpΦpi in ΛpΦp
        fill!(ΛpΦpi, zero(eltype(ΛpΦpi)))
    end
    for ΛmΦmi in ΛmΦm
        fill!(ΛmΦmi, zero(eltype(ΛmΦmi)))
    end
    cache = (; isp, jsp, ism, jsm, Λtemp, σtemp, ΛpΦp, ΛmΦm)
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector}})
    upd_problem = first(b).updatable_problem
    problem = upd_problem.problem
    (; ΛpΦp, ΛmΦm) = cache

    ρs_adjoint = [zeros(num_free_dofs(SpaceModels.material(space_model(problem.model)))) for _ in 1:length(problem.ρp)]
    for i_e in 1:length(problem.ρp)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], upd_problem.ρp_tens, ΛpΦp[i_e] |> collect, true, true)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], upd_problem.ρm_tens, ΛmΦm[i_e] |> collect, true, true)
    end
    return ρs_adjoint
end

function ((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector}})(idx, ψ)
    if !_is_adjoint_vector(b)
        throw(ErrorException("not implemented yet"))
    end
    (; isp, jsp, ism, jsm, Λtemp, σtemp, ΛpΦp, ΛmΦm) = cache

    upd_problem = first(b).updatable_problem
    problem = upd_problem.problem
    T = base_type(architecture(problem))

    Δϵ = T(step(energy_model(problem.model)))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    Λtempp = reshape(@view(Λtemp[1:nxp*nΩp]), (nxp, nΩp))
    Λtempm = reshape(@view(Λtemp[1:nxm*nΩm]), (nxm, nΩm))
    σtempp = Diagonal(@view(σtemp[1:nΩp]))
    σtempm = Diagonal(@view(σtemp[1:nΩm]))

    Φp, Φm = pmview(ψ, problem.model)
    Λ = first(b).cached_solution
    Λ_im2p, Λ_im2m = pmview(Λ[minus½(idx)], problem.model)
    Λ_ip2p, Λ_ip2m = pmview(Λ[plus½(idx)], problem.model)

    for i_e in 1:1:length(problem.ρp)
        s_i = problem.s[i_e, idx]
        τ_i = problem.τ[i_e, idx]

        my_rmul!(σtempp.diag, false)
        for i in 1:size(problem.σ, 2)
            σtempp.diag .+= problem.σ[i_e, i, idx] .* problem.kp[i_e][i].diag
        end

        mul!(Λtempp, Λ_ip2p, σtempp, -T(0.5), false)
        mul!(Λtempp, Λ_im2p, σtempp, -T(0.5), true)
        Λtempp .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2p .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2p

        Sparse3Tensor.special_matmul!(ΛpΦp[i_e], isp, jsp, Λtempp, Φp, Δϵ, true)

        my_rmul!(σtempm.diag, false)
        for i in 1:size(problem.σ, 2)
            σtempm.diag .+= problem.σ[i_e, i, idx] .* problem.km[i_e][i].diag
        end

        mul!(Λtempm, Λ_ip2m, σtempm, -T(0.5), false)
        mul!(Λtempm, Λ_im2m, σtempm, -T(0.5), true)
        Λtempm .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2m .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2m

        Sparse3Tensor.special_matmul!(ΛmΦm[i_e], ism, jsm, Λtempm, Φm, Δϵ, true)
    end
end

# assembly
function initialize_assembly(b::TangentDiscretePNVector)
    upd_problem = b.updatable_problem
    problem = upd_problem.problem
    (n_elem, n_cells) = n_parameters(problem)

    arch = architecture(problem)
    T = base_type(arch)
    ρp_tangent = [similar(upd_problem.ρp_tens.skeleton) |> arch for _ in 1:n_elem]
    ρm_tangent = [similar(upd_problem.ρm_tens.skeleton) |> arch for _ in 1:n_elem]

    onehot = zeros(n_cells)
    for i in 1:n_elem
        if i == b.element_index onehot[b.cell_index] = 1.0 end
        Sparse3Tensor.project!(upd_problem.ρp_tens, onehot)
        Sparse3Tensor.project!(upd_problem.ρm_tens, onehot)
        copyto!(nonzeros(ρp_tangent[i]), nonzeros(upd_problem.ρp_tens.skeleton))
        copyto!(nonzeros(ρm_tangent[i]), nonzeros(upd_problem.ρm_tens.skeleton))
        if i == b.element_index onehot[b.cell_index] = 0.0 end
    end

    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    np, nm = nxp*nΩp, nxm*nΩm
    tmp = allocate_vec(arch, max(np, nm))
    tmp2 = allocate_vec(arch, max(nΩp, nΩm))

    (nd, ne, nσ) = n_sums(problem)
    a = Vector{T}(undef, ne)
    c = [Vector{T}(undef, nσ) for _ in 1:ne]

    cache = (; ρp_tangent, ρm_tangent, a, c, tmp, tmp2)

    return PNVectorAssembler(b, cache)
end

function assemble_at!(rhs, (; b, cache)::PNVectorAssembler{<:TangentDiscretePNVector}, idx, Δ, sym, β=false)
    if _is_adjoint_solution(b.cached_solution) != true
        throw(ErrorException("not implemented yet"))
    end
    upd_problem = b.updatable_problem
    problem = upd_problem.problem

    T = base_type(architecture(problem))
    Δϵ = T(step(energy_model(problem.model)))

    rhsp, rhsm = pmview(rhs, problem.model)
    
    # si = b.problem.s[rhs.i_e, idx]
    # τi = b.problem.τ[rhs.i_e, idx]
    # σi = @view(b.problem.σ[rhs.i_e, :, idx])

    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    (nd, ne, nσ) = n_sums(problem)

    for ie in 1:ne
        cache.a[ie] = b.problem.s[ie, idx]/Δϵ + b.problem.τ[ie, idx]*0.5
        for iσ in 1:nσ
            cache.c[ie][iσ] = -b.problem.σ[ie, iσ, idx]*0.5
        end
    end
    γ = sym ? -1 : 1

    Λp⁻½, Λm⁻½ = pmview(b.cached_solution[minus½(idx)], problem.model)
    Λp⁺½, Λm⁺½ = pmview(b.cached_solution[plus½(idx)], problem.model)

    mul!(@view(rhsp[:]), ZMatrix(cache.ρp_tangent, problem.Ip, problem.kp, cache.a, cache.c, mat_view(cache.tmp, nxp, nΩp), Diagonal(@view(cache.tmp2[1:nΩp]))), @view(Λp⁺½[:]), Δ, β)
    mul!(@view(rhsm[:]), ZMatrix(cache.ρm_tangent, problem.Im, problem.km, cache.a, cache.c, mat_view(cache.tmp, nxm, nΩm), Diagonal(@view(cache.tmp2[1:nΩm]))), @view(Λm⁺½[:]), γ*Δ, β)

    for ie in 1:ne
        cache.a[ie] = -problem.s[ie, idx]/Δϵ + problem.τ[ie, idx]*0.5
        for iσ in 1:nσ
            cache.c[ie][iσ] = -problem.σ[ie, iσ, idx]*0.5
        end
    end
    mul!(@view(rhsp[:]), ZMatrix(cache.ρp_tangent, problem.Ip, problem.kp, cache.a, cache.c, mat_view(cache.tmp, nxp, nΩp), Diagonal(@view(cache.tmp2[1:nΩp]))), @view(Λp⁻½[:]), Δ, true)
    mul!(@view(rhsm[:]), ZMatrix(cache.ρm_tangent, problem.Im, problem.km, cache.a, cache.c, mat_view(cache.tmp, nxm, nΩm), Diagonal(@view(cache.tmp2[1:nΩm]))), @view(Λm⁻½[:]), γ*Δ, true)
    
end

# function Base.getindex(arr::ArrayOfTangentDiscretePNVector, i_e, i_x, Δ=true)
#     system = arr.cached_solution.it.system
#     problem = system.problem
#     # T = base_type(architecture(discrete_system.model))
#     # VT = vec_type(architecture(discrete_system.model))

#     # cv(x) = convert_to_architecture(architecture(discrete_system.model), x)

#     onehot = zeros(num_free_dofs(SpaceModels.material(space_model(problem.model))))
#     onehot[i_x] = Δ
#     Sparse3Tensor.project!(problem.ρp_tens, onehot)
#     Sparse3Tensor.project!(problem.ρm_tens, onehot)

#     copyto!(nonzeros(arr.ρp_tangent), nonzeros(problem.ρp_tens.skeleton))
#     copyto!(nonzeros(arr.ρm_tangent), nonzeros(problem.ρm_tens.skeleton))
#     return TangentDiscretePNVector(arr, i_x, i_e)
#     # return DiscretePNVector(arr.ρp_tangent[i], arr.ρm_tangent[i], arr.cached_solution)
# end
    

# INTEGRATION
# function (arr::ArrayOfTangentDiscretePNVector)(it::DiscretePNIterator)
#     @assert _is_adjoint_solution(it)
#     system = arr.cached_solution.it.system
#     problem = system.problem
#     arch = problem.arch

#     T = base_type(arch)
#     # cv_Int(x) = convert_to_architecture(Int64, architecture(discrete_system.model), x)

#     Δϵ = T(step(energy_model(problem.model)))

#     isp, jsp = arch.(Int64, Sparse3Tensor.get_ijs(problem.ρp_tens))
#     ism, jsm = arch.(Int64, Sparse3Tensor.get_ijs(problem.ρm_tens))

#     (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)

#     Λtemp = allocate_vec(arch, max(nxp*nΩp, nxm*nΩm))
#     Λtempp = reshape(@view(Λtemp[1:nxp*nΩp]), (nxp, nΩp))
#     Λtempm = reshape(@view(Λtemp[1:nxm*nΩm]), (nxm, nΩm))

#     σtemp = allocate_vec(arch, max(nΩp, nΩm))
#     σtempp = Diagonal(@view(σtemp[1:nΩp]))
#     σtempm = Diagonal(@view(σtemp[1:nΩm]))

#     ΛpΦp = [allocate_vec(arch, length(isp)) for _ in 1:length(problem.ρp)]
#     ΛmΦm = [allocate_vec(arch, length(ism)) for _ in 1:length(problem.ρp)]

#     skip_initial = true
#     write_initial = false
#     for (ϵ, i_ϵ) in it
#         if skip_initial
#             skip_initial = false
#             continue
#         end
#         Φp = pview(current_solution(it.system), problem.model)
#         Φm = mview(current_solution(it.system), problem.model)

#         Λ_im2p = pview(arr.cached_solution[i_ϵ], problem.model)
#         Λ_im2m = mview(arr.cached_solution[i_ϵ], problem.model)
#         Λ_ip2p = pview(arr.cached_solution[i_ϵ+1], problem.model)
#         Λ_ip2m = mview(arr.cached_solution[i_ϵ+1], problem.model)

#         for i_e in 1:1:length(problem.ρp)
#             s_i = problem.s[i_e, i_ϵ]
#             τ_i = problem.τ[i_e, i_ϵ]

#             my_rmul!(σtempp.diag, false)
#             for i in 1:size(problem.σ, 2)
#                 σtempp.diag .+= problem.σ[i_e, i, i_ϵ] .* problem.kp[i_e][i].diag
#             end

#             mul!(Λtempp, Λ_ip2p, σtempp, -T(0.5), false)
#             mul!(Λtempp, Λ_im2p, σtempp, -T(0.5), true)
#             Λtempp .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2p .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2p

#             Sparse3Tensor.special_matmul!(ΛpΦp[i_e], isp, jsp, Λtempp, Φp, Δϵ, write_initial)

#             my_rmul!(σtempm.diag, false)
#             for i in 1:size(problem.σ, 2)
#                 σtempm.diag .+= problem.σ[i_e, i, i_ϵ] .* problem.km[i_e][i].diag
#             end

#             mul!(Λtempm, Λ_ip2m, σtempm, -T(0.5), false)
#             mul!(Λtempm, Λ_im2m, σtempm, -T(0.5), true)
#             Λtempm .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2m .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2m

#             Sparse3Tensor.special_matmul!(ΛmΦm[i_e], ism, jsm, Λtempm, Φm, Δϵ, write_initial)
#         end
#         write_initial = true
#     end
#     ρs_adjoint = [zeros(num_free_dofs(SpaceModels.material(space_model(problem.model)))) for _ in 1:length(problem.ρp)]
#     for i_e in 1:length(problem.ρp)
#         Sparse3Tensor.contract!(ρs_adjoint[i_e], problem.ρp_tens, ΛpΦp[i_e] |> collect, true, true)
#         Sparse3Tensor.contract!(ρs_adjoint[i_e], problem.ρm_tens, ΛmΦm[i_e] |> collect, true, true)
#     end
#     return ρs_adjoint
# end

# @concrete struct TangentDiscretePNVector <: AbstractDiscretePNVector
#     parent
#     i_x
#     i_e
# end

# function assemble_rhs!(b, rhs::TangentDiscretePNVector, i, Δ, sym)
#     system = rhs.parent.cached_solution.it.system
#     problem = system.problem
#     arch = system.problem.arch

#     T = base_type(arch)
#     Δϵ = T(step(energy_model(problem.model)))

#     fill!(b, zero(eltype(b)))

#     (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
#     np = nxp*nΩp
#     nm = nxm*nΩm

#     bp = @view(b[1:np])
#     bm = @view(b[np+1:np+nm])

#     si = problem.s[rhs.i_e, i]
#     τi = problem.τ[rhs.i_e, i]
#     σi = problem.σ[rhs.i_e, :, i]

#     Λ_im2 = rhs.parent.cached_solution[i]
#     Λ_ip2 = rhs.parent.cached_solution[i+1]

#     #TODO: move temporary allocations somwhere else
#     tmp = allocate_vec(arch, max(np, nm))
#     tmp2 = allocate_vec(arch, max(nΩp, nΩm))

#     a = [(si/Δϵ + τi*0.5)]
#     c = [(-σi.*0.5)]
#     γ = sym ? -1 : 1

#     mul!(bp, ZMatrix([rhs.parent.ρp_tangent], problem.Ip, [problem.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_ip2[1:np]), Δ, false)
#     mul!(bm, ZMatrix([rhs.parent.ρm_tangent], problem.Im, [problem.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_ip2[np+1:np+nm]), γ*Δ, false)

#     a = [(-si/Δϵ + τi*0.5)]
#     c = [(-σi.*0.5)]
#     mul!(bp, ZMatrix([rhs.parent.ρp_tangent], problem.Ip, [problem.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_im2[1:np]), Δ, true)
#     mul!(bm, ZMatrix([rhs.parent.ρm_tangent], problem.Im, [problem.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_im2[np+1:np+nm]), γ*Δ, true)
# end
