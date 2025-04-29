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
