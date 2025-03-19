function discretize(pn_eq::DegeneratePNEquations, mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    T = base_type(arch)

    n_elem = number_of_elements(pn_eq)

    ## assemble (compute) all the energy matrices
    τ = Vector{T}([absorption_coefficient(pn_eq, e) for e in 1:n_elem])

    space_discretization = discretize_space(pn_eq, mdl, arch)

    return DiscreteDegeneratePNProblem(mdl, arch, direction(pn_eq), τ, space_discretization)
end
