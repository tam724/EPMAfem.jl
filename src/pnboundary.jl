@concrete struct PNSpaceBoundaryCondition
    space_dimension
    space_boundary
    f
end

function discretize(pn_x_bc::PNSpaceBoundaryCondition, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    SM = EPMAfem.SpaceModels
    SM.assemble_linear(SM.∫∂Rp_ngv{typeof(pn_x_bc.space_dimension), typeof(pn_x_bc.space_boundary)}(pn_x_bc.f), space_mdl, SM.even(space_mdl)) |> arch
end


@concrete struct PNDirectionBoundaryCondition
    space_dimension
    space_boundary
    f
end

@concrete struct PNXΩBoundaryCondition
    x_condition
    Ω_condition
end

function discretize()
end
