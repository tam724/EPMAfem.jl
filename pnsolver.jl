
@concrete struct PNGridapModel{S<:DiscreteModel} # this is in fact a grid
    space_model::S
    energy_model
    direction_model
    n_basis::Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}
end

function PNGridapModel(space_model, energy_model, direction_model)
    U, _ = function_spaces(space_model)
    evens = SphericalHarmonicsMatrices.get_even_moments(direction_model, nd(space_model))
    odds = SphericalHarmonicsMatrices.get_odd_moments(direction_model, nd(space_model))

    n_basis = (length(energy_model),
        (num_free_dofs(U[1]), num_free_dofs(U[2])),
        (length(evens), length(odds)))

    return PNGridapModel(
        space_model,
        energy_model,
        direction_model,
        n_basis
    )
end

function space(model::PNGridapModel)
    return model.space_model
end

function energy(model::PNGridapModel)
    return model.energy_model
end

nd(model::PNGridapModel) = nd(space(model))

max_degree(model::PNGridapModel) = model.direction_model
space_directions(model::PNGridapModel) = space_directions(space(model))

function function_spaces(space_model::DiscreteModel)
    V = MultiFieldFESpace([TestFESpace(space_model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1), TestFESpace(space_model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)])
    U = MultiFieldFESpace([TrialFESpace(V[1]), TrialFESpace(V[2])])
    return U, V
end

function number_of_basis_functions(model::PNGridapModel)
    U, _ = function_spaces(space(model))
    x = (p=num_free_dofs(U[1]), m=num_free_dofs(U[2]))

    Ω = (p=length(SphericalHarmonicsMatrices.get_even_moments(max_degree(model), nd(model))),
        m=length(SphericalHarmonicsMatrices.get_odd_moments(max_degree(model), nd(model))))
    return (x=x, Ω=Ω)
end

@inline function pview(v::AbstractVector, model::PNGridapModel)
    (_, (nxp, _), (nΩp, _)) = model.n_basis
    return reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
end

@inline function mview(v::AbstractVector, model::PNGridapModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = model.n_basis
    return reshape(@view(v[nxp*nΩp:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
end

abstract type PNSolver{T} end
