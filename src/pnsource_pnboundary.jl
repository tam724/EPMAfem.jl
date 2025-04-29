@concrete struct PNSpaceBoundaryCondition
    space_dimension
    space_boundary
    f
end

function discretize(pn_x_bc::PNSpaceBoundaryCondition, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    SM = EPMAfem.SpaceModels
    bxp = SM.assemble_linear(SM.∫∂Rp_ngv{typeof(pn_x_bc.space_dimension), typeof(pn_x_bc.space_boundary)}(pn_x_bc.f), space_mdl, SM.even(space_mdl)) |> arch
    bxm = zeros(num_free_dofs(SM.odd(space_mdl))) |> arch
    return bxp, bxm
end


@concrete struct PNDirectionBoundaryCondition
    space_dimension
    space_boundary
    f
end

function discretize(pn_Ω_bc::PNDirectionBoundaryCondition, direction_mdl::SphericalHarmonicsModels.AbstractSphericalHarmonicsModel, arch::PNArchitecture)
    SH = EPMAfem.SphericalHarmonicsModels
    n = EPMAfem.Dimensions.outwards_normal(pn_Ω_bc.space_dimension, pn_Ω_bc.space_boundary, dimensionality(direction_mdl))
    @show n
    n3D = Dimensions.extend_3D(n)
    bΩp = 2 .* SH.assemble_linear(SH.∫S²_nΩgv(n3D, Ω -> pn_Ω_bc.f([Ω...])), direction_mdl, SH.even(direction_mdl), SH.lebedev_quadrature_max()) |> arch
    bΩm = zeros(length(SH.odd(direction_mdl))) |> arch
    return bΩp, bΩm
end

@concrete struct PNSpaceSource
    q
end

function discretize(pn_x_q::PNSpaceSource, space_mdl::SpaceModels.GridapSpaceModel, arch::PNArchitecture)
    SM = EPMAfem.SpaceModels
    bxp = SM.assemble_linear(SM.∫R_μv(pn_x_q.q), space_mdl, SM.even(space_mdl)) |> arch
    bxm = SM.assemble_linear(SM.∫R_μv(pn_x_q.q), space_mdl, SM.odd(space_mdl)) |> arch
    return bxp, bxm
end

@concrete struct PNDirectionSource
    q
end

function discretize(pn_Ω_q::PNDirectionSource, direction_mdl::SphericalHarmonicsModels.AbstractSphericalHarmonicsModel, arch::PNArchitecture)
    SH = EPMAfem.SphericalHarmonicsModels
    bΩp = SH.assemble_linear(SH.∫S²_hv(pn_Ω_q.q), direction_mdl, SH.even(direction_mdl)) |> arch
    bΩm = SH.assemble_linear(SH.∫S²_hv(pn_Ω_q.q), direction_mdl, SH.odd(direction_mdl)) |> arch
    
    return bΩp, bΩm
end

## combined
@concrete struct PNXΩExcitation
    x_excitation
    Ω_excitation
end

function PNXΩBoundaryCondition(space_dimension, space_boundary, fx, fΩ)
    x_condition = PNSpaceBoundaryCondition(space_dimension, space_boundary, fx)
    Ω_condition = PNDirectionBoundaryCondition(space_dimension, space_boundary, fΩ)
    return PNXΩExcitation(x_condition, Ω_condition)
end

function PNXΩSource(qx, qΩ)
    x_source = PNSpaceSource(x -> -qx(x))
    Ω_source = PNDirectionSource(qΩ)
    return PNXΩExcitation(x_source, Ω_source)
end

function discretize(pn_xΩ_bc::PNXΩExcitation, mdl, arch::PNArchitecture)
    space_mdl = space_model(mdl)
    direction_mdl = direction_model(mdl)

    bxp, bxm = discretize(pn_xΩ_bc.x_excitation, space_mdl, arch)
    bΩp, bΩm = discretize(pn_xΩ_bc.Ω_excitation, direction_mdl, arch)
    return (bxp, bxm), (bΩp, bΩm)
end
