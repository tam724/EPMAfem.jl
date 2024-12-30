
function discretize_problem(pn_eq::PNEquations, discrete_model::PNGridapModel, arch::PNArchitecture)
    T = base_type(arch)

    ϵs = energy_model(discrete_model)

    n_elem = number_of_elements(pn_eq)
    n_scat = number_of_scatterings(pn_eq)

    ## assemble (compute) all the energy matrices
    s = Matrix{T}([stopping_power(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([absorption_coefficient(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([scattering_coefficient(pn_eq, e, i, ϵ) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ## instantiate Gridap
    space_mdl = space_model(discrete_model)
    direction_mdl = direction_model(discrete_model)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    ## assemble all the space matrices
    ρp_tens = SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.even(space_mdl), SM.even(space_mdl))
    ρp_tensor = Sparse3Tensor.convert_to_SSM(ρp_tens)
    ρp = [ρp_tensor.skeleton |> arch for _ in 1:number_of_elements(pn_eq)]
    # ρp_skeleton, ρp_projector = SM.build_projector(SM.∫R_uv, space_model, SM.even(space_model), SM.even(space_model))
    # ρp = [SMT((assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[1], V[1]))) for e in 1:number_of_elements(pn_eq)] 
    # ρp = [SMT(ρp_skeleton) for _ in 1:number_of_elements(pn_eq)] 
    # ρp_proj = SMT(ρp_projector)

    ρm_tens = SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.odd(space_mdl))
    ρm_tensor = Sparse3Tensor.convert_to_SSM(ρm_tens)
    ρm = [ρm_tensor.skeleton |> arch for _ in 1:number_of_elements(pn_eq)]
    # ρm_skeleton, ρm_projector = SM.build_projector(SM.∫R_uv, space_model, SM.odd(space_model), SM.odd(space_model))
    # ρm = [Diagonal(VT(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[2], V[2])))) for e in 1:number_of_elements(pn_eq)] 
    # ρm = [Diagonal(VT(diag(ρm_skeleton))) for _ in 1:number_of_elements(pn_eq)]
    # ρm_proj = Diagonal(VT(diag(ρm_projector)))

    ## fill the ρ*s
    # ρ_space = SM.material(space_model)
    ρs = [SM.L2_projection(x -> mass_concentrations(pn_eq, e, x), space_mdl) for e in 1:number_of_elements(pn_eq)]
    for i in 1:number_of_elements(pn_eq)
        Sparse3Tensor.project!(ρp_tensor, ρs[i])
        nonzeros(ρp[i]) .= nonzeros(ρp_tensor.skeleton) |> arch
        Sparse3Tensor.project!(ρm_tensor, ρs[i])
        nonzeros(ρm[i]) .= nonzeros(ρm_tensor.skeleton) |> arch
    end
    # .project_matrices(ρp, ρp_proj, ρs)
    # SM.project_matrices(ρm, ρm_proj, ρs)

    ∂p = [dropzeros(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(discrete_model))] |> arch
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) for ∫ ∈ SM.∫R_u_∂v(dimensionality(discrete_model))] |> arch

    ## assemble all the direction matrices
    # Kpp, Kmm = assemble_scattering_matrices(max_degree(discrete_model), _electron_scattering_kernel(pn_eq, 1, 1), nd(discrete_model))
    kp = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem] |> arch
    km = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem] |> arch

    Ip = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature())) |> arch
    Im = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature())) |> arch

    absΩp_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_absΩuv(dimensionality(discrete_model))]
    absΩp = [BlockedMatrices.blocked_from_mat(absΩp_full[i], SH.get_indices_∫S²absΩuv(direction_mdl)) for (i, dim) in enumerate(dimensions(discrete_model))] |> arch
    # absΩp = absΩp_full |> arch
    Ωpm_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_Ωuv(dimensionality(discrete_model))]
    Ωpm = [BlockedMatrices.blocked_from_mat(Ωpm_full[i], SH.get_indices_∫S²Ωuv(direction_mdl, dim)) for (i, dim) in enumerate(dimensions(discrete_model))] |> arch
    # Ωpm = Ωpm_full |> arch
    DiscretePNProblem(discrete_model, arch, s, τ, σ, ρp, ρp_tensor, ρm, ρm_tensor, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function update_problem!(discrete_system, ρs)
    for (ρp, ρ) in zip(discrete_system.ρp, ρs)
        Sparse3Tensor._project!(ρp, discrete_system.ρp_tens, ρ)
    end
    for (ρm, ρ) in zip(discrete_system.ρm, ρs)
        Sparse3Tensor._project!(ρm, discrete_system.ρm_tens, ρ)
    end
end

function discretize_rhs(pn_ex::PNExcitation, discrete_model::PNGridapModel, arch::PNArchitecture)
    T = base_type(arch)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = space_model(discrete_model)
    direction_mdl = direction_model(discrete_model)
    ## assemble excitation 
    gϵs = [Vector{T}([beam_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy_model(discrete_model)]) for i in 1:number_of_beam_energies(pn_ex)]
    gxps = [SM.assemble_linear(SM.∫∂R_ngv{Dimensions.Z}(x -> beam_space_distribution(pn_ex, i, Dimensions.extend_3D(x))), space_mdl, SM.even(space_mdl)) for i in 1:number_of_beam_positions(pn_ex)] |> arch
    nz = Dimensions.cartesian_unit_vector(Dimensions.Z(), dimensionality(discrete_model))
    nz3D = Dimensions.extend_3D(nz)
    gΩps = [SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> beam_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl)) for i in 1:number_of_beam_directions(pn_ex)] |> arch
    return ArrayOfRank1DiscretePNVector{false}(discrete_model, arch, gϵs, gxps, gΩps)
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

function discretize_extraction(pn_ex::PNExtraction, discrete_model::PNGridapModel, arch::PNArchitecture)
    T = base_type(arch)

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = space_model(discrete_model)
    direction_mdl = direction_model(discrete_model)

    ## ... and extraction
    μϵs = [Vector{T}([extraction_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy_model(discrete_model)]) for i in 1:number_of_extractions(pn_ex)]
    μxps = [SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.even(space_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch
    μΩps = [SH.assemble_linear(SH.∫S²_hv(Ω -> extraction_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl)) for i in 1:number_of_extractions(pn_ex)] |> arch

    return VecOfRank1DiscretePNVector{true}(discrete_model, arch, μϵs, μxps, μΩps)
end