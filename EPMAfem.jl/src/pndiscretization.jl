
function discretize_problem(pn_eq::PNEquations, discrete_model::PNGridapModel)
    MT = mat_type(architecture(discrete_model))
    VT = vec_type(architecture(discrete_model))
    SMT = smat_type(architecture(discrete_model))
    T = base_type(discrete_model)

    ϵs = energy(discrete_model)

    n_elem = number_of_elements(pn_eq)
    n_scat = number_of_scatterings(pn_eq)

    ## assemble (compute) all the energy matrices
    s = Matrix{T}([stopping_power(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    τ = Matrix{T}([absorption_coefficient(pn_eq, e, ϵ) for e in 1:n_elem, ϵ ∈ ϵs])
    σ = Array{T}([scattering_coefficient(pn_eq, e, i, ϵ) for e in 1:n_elem, i in 1:n_scat, ϵ ∈ ϵs])

    ## instantiate Gridap
    space_model = space(discrete_model)
    direction_model = direction(discrete_model)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    ## assemble all the space matrices
    ρp_skeleton, ρp_projector = SM.build_projector(SM.∫R_uv, space_model, SM.even(space_model), SM.even(space_model))
    # ρp = [SMT((assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[1], V[1]))) for e in 1:number_of_elements(pn_eq)] 
    ρp = [SMT(ρp_skeleton) for _ in 1:number_of_elements(pn_eq)] 
    ρp_proj = SMT(ρp_projector)

    ρm_skeleton, ρm_projector = SM.build_projector(SM.∫R_uv, space_model, SM.odd(space_model), SM.odd(space_model))
    # ρm = [Diagonal(VT(diag(assemble_bilinear(∫ρuv, (_mass_concentrations(pn_eq, e), gap_model), U[2], V[2])))) for e in 1:number_of_elements(pn_eq)] 
    ρm = [Diagonal(VT(diag(ρm_skeleton))) for _ in 1:number_of_elements(pn_eq)]
    ρm_proj = Diagonal(VT(diag(ρm_projector)))

    ## fill the ρ*s
    # ρ_space = SM.material(space_model)
    ρs = [VT(SM.L2_projection(x -> mass_concentrations(pn_eq, e, x), space_model)) for e in 1:number_of_elements(pn_eq)]
    SM.project_matrices(ρp, ρp_proj, ρs)
    SM.project_matrices(ρm, ρm_proj, ρs)

    ∂p = [SMT(dropzeros(SM.assemble_bilinear(∫, space_model, SM.even(space_model), SM.even(space_model)))) for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(discrete_model))]
    ∇pm = [SMT(SM.assemble_bilinear(∫, space_model, SM.odd(space_model), SM.even(space_model))) for ∫ ∈ SM.∫R_u_∂v(dimensionality(discrete_model))]

    ## assemble all the direction matrices
    # Kpp, Kmm = assemble_scattering_matrices(max_degree(discrete_model), _electron_scattering_kernel(pn_eq, 1, 1), nd(discrete_model))
    kp = [[to_diag(VT, SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_model, SH.even(direction_model), SH.even(direction_model), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem]
    km = [[to_diag(VT, SH.assemble_bilinear(SH.∫S²_kuv(μ->electron_scattering_kernel(pn_eq, e, i, μ)), direction_model, SH.odd(direction_model), SH.odd(direction_model), SH.hcubature_quadrature(1e-9, 1e-9))) for i in 1:n_scat] for e in 1:n_elem]

    Ip = to_diag(VT, SH.assemble_bilinear(SH.∫S²_uv, direction_model, SH.even(direction_model), SH.even(direction_model), SH.exact_quadrature()))
    Im = to_diag(VT, SH.assemble_bilinear(SH.∫S²_uv, direction_model, SH.odd(direction_model), SH.odd(direction_model), SH.exact_quadrature()))

    absΩp = [MT(SH.assemble_bilinear(∫, direction_model, SH.even(direction_model), SH.even(direction_model), SH.exact_quadrature())) for ∫ ∈ SH.∫S²_absΩuv(dimensionality(discrete_model))]
    Ωpm = [MT(SH.assemble_bilinear(∫, direction_model, SH.even(direction_model), SH.odd(direction_model), SH.exact_quadrature())) for ∫ ∈ SH.∫S²_Ωuv(dimensionality(discrete_model))]

    DiscretePNSystem(discrete_model, s, τ, σ, ρp, ρp_proj, ρm, ρm_proj, ∂p, ∇pm, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_rhs(pn_ex::PNExcitation, discrete_model::PNGridapModel)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_model = space(discrete_model)
    direction_model = direction(discrete_model)
    ## assemble excitation 
    gϵs = [Vector{T}([beam_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy(discrete_model)]) for i in 1:number_of_beam_energies(pn_ex)]
    gxps = [VT(SM.assemble_linear(SM.∫∂R_ngv{Dimensions.Z}(x -> beam_space_distribution(pn_ex, i, Dimensions.extend_3D(x))), space_model, SM.even(space_model))) for i in 1:number_of_beam_positions(pn_ex)]
    nz = Dimensions.cartesian_unit_vector(Dimensions.Z(), dimensionality(discrete_model))
    nz3D = Dimensions.extend_3D(nz)
    gΩps = [VT(SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> beam_direction_distribution(pn_ex, i, Ω)), direction_model, SH.even(direction_model))) for i in 1:number_of_beam_directions(pn_ex)]
    return ArrayOfRank1DiscretePNVector{false}(discrete_model, gϵs, gxps, gΩps)
end

function discretize_extraction(pn_ex::PNExtraction, discrete_model::PNGridapModel)
    VT = vec_type(discrete_model)
    T = base_type(discrete_model)

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_model = space(discrete_model)
    direction_model = direction(discrete_model)

    ## ... and extraction
    μϵs = [Vector{T}([extraction_energy_distribution(pn_ex, i, ϵ) for ϵ ∈ energy(discrete_model)]) for i in 1:number_of_extractions(pn_ex)]
    μxps = [VT(SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_model, SM.even(space_model))) for i in 1:number_of_extractions(pn_ex)]
    μΩps = [VT(SH.assemble_linear(SH.∫S²_hv(Ω -> extraction_direction_distribution(pn_ex, i, Ω)), direction_model, SH.even(direction_model))) for i in 1:number_of_extractions(pn_ex)]

    return VecOfRank1DiscretePNVector{true}(discrete_model, μϵs, μxps, μΩps)
end

# function discretize_rhs(pn_eq::PNEquations, discrete_model::PNGridapModel)



#     return VecOfRank1DiscretePNVector{true}(discrete_model, )