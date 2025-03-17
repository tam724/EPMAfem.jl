function discretize_space(pn_eq::AbstractMonochromPNEquations, mdl::DiscreteMonochromPNModel, arch::PNArchitecture)
    n_elem = number_of_elements(pn_eq)

    space_mdl = space_model(mdl)
    SM = EPMAfem.SpaceModels

    ρp = [SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.even(space_mdl), SM.even(space_mdl)) |> arch for e in 1:n_elem]
    ρm = [Diagonal(Vector(diag(SM.assemble_bilinear(SM.∫R_ρuv(x -> mass_concentrations(pn_eq, e, x)), space_mdl, SM.odd(space_mdl), SM.odd(space_mdl))))) |> arch for e in 1:n_elem]

    ∂p = [dropzeros!(SM.assemble_bilinear(∫, space_mdl, SM.even(space_mdl), SM.even(space_mdl))) for ∫ ∈ SM.∫∂R_absn_uv(dimensionality(mdl))] |> arch
    ∇pm = [SM.assemble_bilinear(∫, space_mdl, SM.odd(space_mdl), SM.even(space_mdl)) for ∫ ∈ SM.∫R_u_∂v(dimensionality(mdl))] |> arch

    return SpaceDiscretization(space_mdl, arch, ρp, ρm, ∂p, ∇pm)
end

function discretize_direction(pn_eq::AbstractMonochromPNEquations, mdl::DiscreteMonochromPNModel, arch::PNArchitecture)
    direction_mdl = direction_model(mdl)
    SH = EPMAfem.SphericalHarmonicsModels

    n_elem = number_of_elements(pn_eq)

    ## assemble all the direction matrices
    kp = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9)))] for e in 1:n_elem] |> arch
    km = [[to_diag(SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel(pn_eq, e)), direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9)))] for e in 1:n_elem] |> arch

    Ip = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.exact_quadrature())) |> arch
    Im = to_diag(SH.assemble_bilinear(SH.∫S²_uv, direction_mdl, SH.odd(direction_mdl), SH.odd(direction_mdl), SH.exact_quadrature())) |> arch

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

    space_discretization = discretize_space(pn_eq, mdl, arch)
    direction_discretization = discretize_direction(pn_eq, mdl, arch)

    return DiscreteMonochromPNProblem(mdl, arch, τ, σ, space_discretization, direction_discretization)
end