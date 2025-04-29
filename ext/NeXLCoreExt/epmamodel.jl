function epma_model(eq::EPMAEquations, space_extents, num_cells, PN_N)
    # sanity check (first dimension should be depth, material in negative z direction)
    @assert space_extents[1] < 0u"m"
    # sanity check (first dimension should be depth, surface at z = 0)
    @assert space_extents[2] == 0u"m"

    space_extents_dimless = dimless.(space_extents, eq.dim_basis)
    space_model = SpaceModels.GridapSpaceModel(CartesianDiscreteModel(space_extents_dimless, num_cells))
    energy_model = eq.energy_model_dimless
    direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(PN_N, EPMAfem.Dimensions.dimensionality(space_model))
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
    return model
end

