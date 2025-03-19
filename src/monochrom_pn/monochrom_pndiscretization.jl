
function discretize_space(pn_eq::Union{AbstractMonochromPNEquations, AbstractDegeneratePNEquations}, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    n_elem = number_of_elements(pn_eq)

    SM = EPMAfem.SpaceModels

    ρp = [diag_if_diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.even(space_mdl), SM.even(space_mdl))) |> arch for e in 1:n_elem]
    ρm = [diag_if_diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl,  SM.odd(space_mdl),  SM.odd(space_mdl))) |> arch for e in 1:n_elem]

    ∂p = [dropzeros!(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) for ∫ ∈ SM.∫∂R_absn_uv(SM.dimensionality(space_mdl))] |> arch
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) for ∫ ∈ SM.∫R_u_∂v(SM.dimensionality(space_mdl))] |> arch

    return SpaceDiscretization(space_mdl, arch, ρp, ρm, ∂p, ∇pm)
end

function discretize_direction(pn_eq::AbstractMonochromPNEquations, direction_mdl::SphericalHarmonicsModels.AbstractSphericalHarmonicsModel, arch::PNArchitecture)
    SH = EPMAfem.SphericalHarmonicsModels

    n_elem = number_of_elements(pn_eq)

    ## assemble all the direction matrices
    kp = [[diag_if_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9)))] for e in 1:n_elem] |> arch
    km = [[diag_if_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e)), direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9)))] for e in 1:n_elem] |> arch

    Ip = diag_if_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature())) |> arch
    Im = diag_if_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature())) |> arch

    absΩp_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_absΩuv(dimensionality(mdl))]
    absΩp = [BlockedMatrices.blocked_from_mat(absΩp_full[i], SH.get_indices_∫S²absΩuv(direction_mdl)) for (i, dim) in enumerate(dimensions(mdl))] |> arch

    Ωpm_full = [SH.assemble_bilinear(∫, direction_mdl, SH.even(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature()) for ∫ ∈ SH.∫S²_Ωuv(dimensionality(mdl))]
    Ωpm = [BlockedMatrices.blocked_from_mat(Ωpm_full[i], SH.get_indices_∫S²Ωuv(direction_mdl, dim)) for (i, dim) in enumerate(dimensions(mdl))] |> arch

    return DirectionDiscretization(direction_mdl, arch, Ip, Im, kp, km, absΩp, Ωpm)
end

function discretize_problem(pn_eq::AbstractMonochromPNEquations, mdl::DiscreteMonochromPNModel, arch::PNArchitecture; updatable=false)
    T = base_type(arch)

    n_elem = number_of_elements(pn_eq)

    ## assemble (compute) all the energy matrices
    τ = Vector{T}([absorption_coefficient(pn_eq, e) for e in 1:n_elem])
    σ = Vector{T}([scattering_coefficient(pn_eq, e) for e in 1:n_elem])

    space_mdl = space_model(mdl)
    space_discretization = discretize_space(pn_eq, space_mdl, arch)
    direction_mdl = direction_model(mdl)
    direction_discretization = discretize_direction(pn_eq, direction_mdl, arch)

    return DiscreteMonochromPNProblem(mdl, arch, τ, σ, space_discretization, direction_discretization)
end

function discretize_rhs(b::PNXΩExcitation, mdl::DiscreteMonochromPNModel, arch::PNArchitecture)
    (bxp, _), (bΩp, _) = discretize(b, mdl, arch)
    return DiscreteMonochromPNVector(false, mdl, arch, bxp, bΩp)
end
