function discretize_mass_concentrations(mass_concentrations::AbstractVector{<:Function}, space_mdl::SpaceModels.GridapSpaceModel)
    SM = EPMAfem.SpaceModels
    return vcat((SM.L2_projection(x -> mass_concentrations[e](x), space_mdl)' for e in 1:length(mass_concentrations))...)
end

function discretize_mass_concentrations(pn_eq::AbstractPNEquations, mdl::DiscretePNModel)
    space_mdl = space_model(mdl)
    discretize_mass_concentrations([x -> mass_concentrations(pn_eq, e, x) for e in 1:number_of_elements(pn_eq)], space_mdl)
end

function discretize_mass_concentrations(mass_concentrations::AbstractVector{<:Function}, mdl::DiscretePNModel)
    space_mdl = space_model(mdl)
    discretize_mass_concentrations(mass_concentrations, space_mdl)
end

@concrete struct SpaceDiscretization
    space_model
    arch

    ρp
    ρm

    ∂p
    ∇pm
end

lazy_space_matrices(space_discretization::SpaceDiscretization) = lazy.(space_discretization.ρp), lazy.(space_discretization.ρm), lazy.(space_discretization.∂p), lazy.(space_discretization.∇pm)


@concrete struct DirectionDiscretization
    direction_model
    arch

    Ip
    Im
    kp
    km
    absΩp
    Ωpm
end

lazy_direction_matrices(direction_discretization::DirectionDiscretization) = lazy(direction_discretization.Ip), lazy(direction_discretization.Im), direction_discretization.kp .|> x -> lazy.(x), direction_discretization.km .|> x -> lazy.(x), lazy.(direction_discretization.absΩp), lazy.(direction_discretization.Ωpm)



function discretize_space(pn_eq::Union{AbstractPNEquations, AbstractMonochromPNEquations, AbstractDegeneratePNEquations}, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    n_elem = number_of_elements(pn_eq)

    SM = EPMAfem.SpaceModels

    ρp = [diag_if_diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.even(space_mdl), SM.even(space_mdl))) |> arch for e in 1:n_elem]
    ρm = [diag_if_diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl,  SM.odd(space_mdl),  SM.odd(space_mdl))) |> arch for e in 1:n_elem]

    ∂p = [dropzeros!(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) |> arch for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(space_mdl))] 
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) |> arch for ∫ ∈ SM.∫R_u_∂v(dimensionality(space_mdl))] 

    return SpaceDiscretization(space_mdl, arch, ρp, ρm, ∂p, ∇pm)
end

function discretize_space_updatable(pn_eq::Union{AbstractPNEquations, AbstractMonochromPNEquations, AbstractDegeneratePNEquations}, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    n_elem = number_of_elements(pn_eq)

    SM = EPMAfem.SpaceModels

    ρp_tens = Sparse3Tensor.convert_to_SSM(SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.even(space_mdl), SM.even(space_mdl)))
    ρp = [similar(ρp_tens.skeleton) |> arch for _ in 1:n_elem]

    ρm_tens = Sparse3Tensor.convert_to_SSM(SM.assemble_trilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.odd(space_mdl)))
    ρm = [similar(ρm_tens.skeleton) |> arch for _ in 1:n_elem]


    ∂p = [dropzeros!(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(space_mdl))] |> arch
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) for ∫ ∈ SM.∫R_u_∂v(dimensionality(space_mdl))] |> arch

    return SpaceDiscretization(space_mdl, arch, ρp, ρm, ∂p, ∇pm), ρp_tens, ρm_tens
end

function discretize_direction(pn_eq::Union{AbstractPNEquations, AbstractMonochromPNEquations}, direction_mdl::SphericalHarmonicsModels.AbstractSphericalHarmonicsModel, arch::PNArchitecture)
    SH = EPMAfem.SphericalHarmonicsModels

    n_elem = number_of_elements(pn_eq)
    n_scat = number_of_scatterings(pn_eq)

    ## assemble all the direction matrices
    kp = [[diag_if_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e, i)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) |> arch for i in 1:n_scat] for e in 1:n_elem]
    km = [[diag_if_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e, i)), direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))) |> arch for i in 1:n_scat] for e in 1:n_elem]

    Ip = diag_if_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature())) |> arch
    Im = diag_if_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature())) |> arch

    absΩp_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_absΩuv(dimensionality(direction_mdl))]
    if direction_mdl isa SH.EOSphericalHarmonicsModel
        absΩp = arch.(absΩp_full)
    else
        @assert direction_mdl isa SH.EEEOSphericalHarmonicsModel
        absΩp = [BlockedMatrices.blocked_from_mat(absΩp_full[i], SH.get_indices_∫S²absΩuv(direction_mdl)) |> arch for (i, dim) in enumerate(dimensions(dimensionality(direction_mdl)))] 
    end

    Ωpm_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_Ωuv(dimensionality(direction_mdl))]
    if direction_mdl isa SH.EOSphericalHarmonicsModel
        Ωpm = arch.(Ωpm_full)
    else
        @assert direction_mdl isa SH.EEEOSphericalHarmonicsModel
        Ωpm = [BlockedMatrices.blocked_from_mat(Ωpm_full[i], SH.get_indices_∫S²Ωuv(direction_mdl, dim)) |> arch for (i, dim) in enumerate(dimensions(dimensionality(direction_mdl)))] 
    end

    return DirectionDiscretization(direction_mdl, arch, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_problem(pn_eq::AbstractPNEquations, mdl::DiscretePNModel, arch::PNArchitecture; updatable=false)
    T = base_type(arch)

    ϵs = energy_model(mdl)

    # n_scat = number_of_scatterings(pn_eq)

    ## assemble (compute) all the energy matrices
    s = Matrix{T}(stopping_power(pn_eq, ϵs))
    τ = Matrix{T}(absorption_coefficient(pn_eq, ϵs))
    σ = Array{T}(scattering_coefficient(pn_eq, ϵs))

    space_mdl = space_model(mdl)
    if updatable
        space_discretization, ρp_tens, ρm_tens = discretize_space_updatable(pn_eq, space_mdl, arch)
    else
        space_discretization = discretize_space(pn_eq, space_mdl, arch)
    end

    ## assemble all the direction matrices
    direction_mdl = direction_model(mdl)
    direction_discretization = discretize_direction(pn_eq, direction_mdl, arch)

    problem = DiscretePNProblem(mdl, arch, s, τ, σ, space_discretization, direction_discretization)
    if updatable
        n_parameters = (number_of_elements(pn_eq), n_basis(mdl).nx.m)
        upd_problem = UpdatableDiscretePNProblem(problem, ρp_tens, ρm_tens, n_parameters)
        
        ρs = discretize_mass_concentrations(pn_eq, mdl)
        update_problem!(upd_problem, ρs)
        return upd_problem
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
    gxps = [sparse(SM.assemble_linear(SM.∫∂R_ngv{Dimensions.Z}(x -> beam_space_distribution(pn_ex, i, Dimensions.extend_3D(x))), space_mdl, SM.even(space_mdl))) |> arch for i in 1:number_of_beam_positions(pn_ex)]

    nz = Dimensions.cartesian_unit_vector(Dimensions.Z(), dimensionality(mdl))
    nz3D = Dimensions.extend_3D(nz)
    gΩps = [SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> beam_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl), SH.lebedev_quadrature_max()) |> arch for i in 1:number_of_beam_directions(pn_ex)]
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
    μΩps = [SH.assemble_linear(SH.∫S²_hv(Ω -> extraction_direction_distribution(pn_ex, i, Ω)), direction_mdl, SH.even(direction_mdl)) |> arch for i in 1:number_of_extractions(pn_ex)]

    if updatable
        ρ_proj = SM.assemble_bilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.even(space_mdl))
        ρs = discretize_mass_concentrations(pn_ex.pn_eq, mdl)
        n_parameters = (number_of_elements(pn_ex.pn_eq), n_basis(mdl).nx.m)
        return [UpdatableRank1DiscretePNVector(Rank1DiscretePNVector(true, mdl, arch, μϵs[i], ρ_proj*@view(ρs[i, :]) |> arch, μΩps[i]), EPMAfem.PNNoAbsorption(mdl, arch, ρ_proj, i), n_parameters) for i in 1:number_of_extractions(pn_ex)]
    else
        μxps = [SM.assemble_linear(SM.∫R_μv(x -> extraction_space_distribution(pn_ex, i, x)), space_mdl, SM.even(space_mdl)) |> arch for i in 1:number_of_extractions(pn_ex)]
        return [Rank1DiscretePNVector(true, mdl, arch, μϵs[i], μxps[i], μΩps[i]) for i in 1:number_of_extractions(pn_ex)]
    end
end

function discretize_outflux(mdl::DiscretePNModel, arch::PNArchitecture, ϵ_func=ϵ->one(base_type(arch)))
    T = base_type(arch)

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = space_model(mdl)
    direction_mdl = direction_model(mdl)

    ## ... and extraction
    μϵ = Vector{T}([ϵ_func(ϵ) for ϵ ∈ energy_model(mdl)])
    n = VectorValue(1.0, 0.0, 0.0)
    μΩp = SH.assemble_linear(SH.∫S²_hv(Ω -> abs(dot(Ω, n))), direction_mdl, SH.even(direction_mdl)) |> arch
    μxp = SM.assemble_linear(SM.∫∂R_ngv{Dimensions.Z}(x -> isapprox(x[1], 0.0, atol=1e-12)), space_mdl, SM.even(space_mdl)) |> arch
    return Rank1DiscretePNVector(true, mdl, arch, μϵ, μxp, μΩp)
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
