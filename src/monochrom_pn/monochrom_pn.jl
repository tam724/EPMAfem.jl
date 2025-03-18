# This basically mirrors the types from the large pn-system, but without energy dependence
# will try to reuse as much of the functionality from the large system ..

abstract type AbstractMonochromPNEquations end

@concrete struct DiscreteMonochromPNModel
    space_mdl
    direction_mdl
    number_of_basis_functions::@NamedTuple{nx::@NamedTuple{p::Int64, m::Int64}, nΩ::@NamedTuple{p::Int64, m::Int64}}
end

function DiscreteMonochromPNModel(space_model, direction_model, allow_different_dimensionality=false)
    if !allow_different_dimensionality @assert SpaceModels.dimensionality(space_model) == SphericalHarmonicsModels.dimensionality(direction_model) end
    n_basis_space = SpaceModels.n_basis(space_model)
    n_basis_direction = SphericalHarmonicsModels.n_basis(direction_model)

    number_of_basis_functions = (
        nx = n_basis_space,
        nΩ = n_basis_direction)

    return DiscreteMonochromPNModel(
        space_model,
        direction_model,
        number_of_basis_functions
    )
end

Base.show(io::IO, m::DiscreteMonochromPNModel) = print(io, "DiscreteMonochromPNModel [$(length(dimensions(m)))D, N=$(direction_model(m).N), cells:$(n_basis(m).nx.m)]")
Base.show(io::IO, ::MIME"text/plain", m::DiscreteMonochromPNModel) = show(io, m)

function space_model(model::DiscreteMonochromPNModel)
    return model.space_mdl
end

function direction_model(model::DiscreteMonochromPNModel)
    return model.direction_mdl
end

dimensionality(model::DiscreteMonochromPNModel) = SpaceModels.dimensionality(space_model(model))
dimensions(model::DiscreteMonochromPNModel) = Dimensions.dimensions(dimensionality(model))

n_basis(model::DiscreteMonochromPNModel) = model.number_of_basis_functions

function pview(v::AbstractVector, model::DiscreteMonochromPNModel)
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    return reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
end

function mview(v::AbstractVector, model::DiscreteMonochromPNModel)
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    return reshape(@view(v[nxp*nΩp+1:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
end

function pmview(v::AbstractVector, model::DiscreteMonochromPNModel)
    ((nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    pview = reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
    mview = reshape(@view(v[nxp*nΩp+1:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
    return pview, mview
end

include("monochrom_pnequations.jl")
include("monochrom_pnproblem.jl")
include("monochrom_pnvector.jl")
include("monochrom_pndiscretization.jl")
