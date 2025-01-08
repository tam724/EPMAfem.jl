function discretize_mass_concentrations(pn_eq::AbstractPNEquations, mdl::DiscretePNModel)
    space_mdl = space_model(mdl)
    SM = EPMAfem.SpaceModels
    return vcat((SM.L2_projection(x -> mass_concentrations(pn_eq, e, x), space_mdl)' for e in 1:number_of_elements(pn_eq))...)
end

function discretize_problem(pn_eq::AbstractPNEquations, mdl::DiscretePNModel, arch::PNArchitecture; updatable=false)
    T = base_type(arch)

    ϵs = energy_model(mdl)

    n_elem = number_of_elements(pn_eq)
    n_scat = number_of_scatterings(pn_eq)

    ## assemble (compute) all the energy matrices
    s = Matrix{T}([stopping_power(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([absorption_coefficient(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([scattering_coefficient(pn_eq, e, i, ϵ) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ## instantiate Gridap
    space_mdl = space_model(mdl)
    direction_mdl = direction_model(mdl)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    ## assemble all the space matrices
    if updatable
        ρp_tens = Sparse3Tensor.convert_to_SSM(SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.even(space_mdl), SM.even(space_mdl)))
        ρp = [similar(ρp_tens.skeleton) |> arch for _ in 1:n_elem]

        ρm_tens = Sparse3Tensor.convert_to_SSM(SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.odd(space_mdl)))
        ρm = [similar(ρm_tens.skeleton) |> arch for _ in 1:n_elem]

        ρs = discretize_mass_concentrations(pn_eq, mdl)
        for i in 1:number_of_elements(pn_eq)
            Sparse3Tensor.project!(ρp_tens, @view(ρs[i, :]))
            nonzeros(ρp[i]) .= nonzeros(ρp_tens.skeleton) |> arch
            Sparse3Tensor.project!(ρm_tens, @view(ρs[i, :]))
            nonzeros(ρm[i]) .= nonzeros(ρm_tens.skeleton) |> arch
        end
    else
        ρp = [SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.even(space_mdl), SM.even(space_mdl)) |> arch for e in 1:number_of_elements(pn_eq)]
        ρm = [Diagonal(Vector(diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.odd(space_mdl), SM.odd(space_mdl))))) |> arch for e in 1:number_of_elements(pn_eq)]
    end

    ∂p = [dropzeros!(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(mdl))] |> arch
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) for ∫ ∈ SM.∫R_u_∂v(dimensionality(mdl))] |> arch

    ## assemble all the direction matrices
    # Kpp, Kmm = assemble_scattering_matrices(max_degree(discrete_model), _electron_scattering_kernel(pn_eq, 1, 1), nd(discrete_model))
    kp = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem] |> arch
    km = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem] |> arch

    Ip = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature())) |> arch
    Im = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature())) |> arch

    absΩp_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_absΩuv(dimensionality(mdl))]
    absΩp = [BlockedMatrices.blocked_from_mat(absΩp_full[i], SH.get_indices_∫S²absΩuv(direction_mdl)) for (i, dim) in enumerate(dimensions(mdl))] |> arch
    # absΩp = absΩp_full |> arch
    Ωpm_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_Ωuv(dimensionality(mdl))]
    Ωpm = [BlockedMatrices.blocked_from_mat(Ωpm_full[i], SH.get_indices_∫S²Ωuv(direction_mdl, dim)) for (i, dim) in enumerate(dimensions(mdl))] |> arch
    # Ωpm = Ωpm_full |> arch
    problem = DiscretePNProblem(mdl, arch, s, τ, σ, ρp, ρm, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
    if updatable
        n_parameters = (number_of_elements(pn_eq), n_basis(mdl).nx.m)
        return UpdatableDiscretePNProblem(problem, ρp_tens, ρm_tens, n_parameters)
    else
        return problem
    end
end

function discretize_rhs(pn_ex::PNExcitation, mdl::DiscretePNModel, arch::PNArchitecture)
    T = base_type(arch)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = space_model(mdl)
    direction_mdl = direction_model(mdl)
    ## assemble excitation 
    gϵs = [Vector{T}([beam_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy_model(mdl)]) for i in 1:number_of_beam_energies(pn_ex)]
    gxps = [SM.assemble_linear(SM.∫∂R_ngv{Dimensions.Z}(x -> beam_space_distribution(pn_ex, i, Dimensions.extend_3D(x))), space_mdl, SM.even(space_mdl)) for i in 1:number_of_beam_positions(pn_ex)] |> arch
    nz = Dimensions.cartesian_unit_vector(Dimensions.Z(), dimensionality(mdl))
    nz3D = Dimensions.extend_3D(nz)
    gΩps = [SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> beam_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl), SH.lebedev_quadrature_max()) for i in 1:number_of_beam_directions(pn_ex)] |> arch
    return [Rank1DiscretePNVector(false, mdl, arch, gϵs[i], gxps[j], gΩps[k]) for i in 1:number_of_beam_energies(pn_ex), j in 1:number_of_beam_positions(pn_ex), k in 1:number_of_beam_directions(pn_ex)]
end

# function discretize_stange_rhs(pn_ex::PNExcitation, discrete_model::PNGridapModel, arch::PNArchitecture)
#     T = base_type(arch)

#     SM = EPMAfem.SpaceModels
#     SH = EPMAfem.SphericalHarmonicsModels

#     space_mdl = space_model(discrete_model)
#     direction_mdl = direction_model(discrete_model)
#     ## assemble excitation 
#     gϵs = Vector{T}([beam_energy_distribution(pn_ex, 1, ϵ) for ϵ ∈ energy_model(discrete_model)])
#     gxps = (SM.assemble_linear(SM.∫R_μv(x -> -exp(-100.0*((x[1]+0.5)^2+(x[2])^2))), space_mdl, SM.even(space_mdl)))
#     gΩps = (SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_mdl, SH.even(direction_mdl)))
#     return Rank1DiscretePNVector{false}(discrete_model, gϵs, gxps, gΩps)
# end

function discretize_extraction(pn_ex::PNExtraction, mdl::DiscretePNModel, arch::PNArchitecture; updatable=true)
    T = base_type(arch)

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = space_model(mdl)
    direction_mdl = direction_model(mdl)

    ## ... and extraction
    μϵs = [Vector{T}([extraction_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy_model(mdl)]) for i in 1:number_of_extractions(pn_ex)]
    μΩps = [SH.assemble_linear(SH.∫S²_hv(Ω -> extraction_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch

    if updatable
        ρ_proj = SM.assemble_bilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.even(space_mdl))
        ρs = discretize_mass_concentrations(pn_ex.pn_eq, mdl)
        n_parameters = (number_of_elements(pn_ex.pn_eq), n_basis(mdl).nx.m)
        return [UpdatableRank1DiscretePNVector(Rank1DiscretePNVector(true, mdl, arch, μϵs[i], ρ_proj*@view(ρs[i, :]) |> arch, μΩps[i]), ρ_proj, n_parameters, i) for i in 1:number_of_extractions(pn_ex)]
    else
        μxps = [SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.even(space_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch
        return [Rank1DiscretePNVector(true, mdl, arch, μϵs[i], μxps[i], μΩps[i]) for i in 1:number_of_extractions(pn_ex)]
    end
end

# function discretize_extraction_old(pn_ex::PNExtraction, discrete_model::PNGridapModel, arch::PNArchitecture)
#     T = base_type(arch)

#     ## instantiate Gridap
#     SM = EPMAfem.SpaceModels
#     SH = EPMAfem.SphericalHarmonicsModels

#     space_mdl = space_model(discrete_model)
#     direction_mdl = direction_model(discrete_model)

#     ## ... and extraction
#     μϵs = [Vector{T}([extraction_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy_model(discrete_model)]) for i in 1:number_of_extractions(pn_ex)]
#     μxps = [SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.even(space_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch
#     μΩps = [SH.assemble_linear(SH.∫S²_hv(Ω -> extraction_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch

#     return VecOfRank1DiscretePNVector{true}(discrete_model, arch, μϵs, μxps, μΩps)
# end