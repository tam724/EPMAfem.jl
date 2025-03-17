@concrete struct DiscretePNModel
    space_mdl
    energy_mdl
    direction_mdl
    number_of_basis_functions::@NamedTuple{nϵ::Int64, nx::@NamedTuple{p::Int64, m::Int64}, nΩ::@NamedTuple{p::Int64, m::Int64}}
end

function DiscretePNModel(space_model, energy_model, direction_model, allow_different_dimensionality=false)
    if !allow_different_dimensionality @assert SpaceModels.dimensionality(space_model) == SphericalHarmonicsModels.dimensionality(direction_model) end
    n_basis_energy = length(energy_model)
    n_basis_space = SpaceModels.n_basis(space_model)
    n_basis_direction = SphericalHarmonicsModels.n_basis(direction_model)

    number_of_basis_functions = (nϵ = n_basis_energy,
        nx = n_basis_space,
        nΩ = n_basis_direction)

    return DiscretePNModel(
        space_model,
        energy_model,
        direction_model,
        number_of_basis_functions
    )
end

Base.show(io::IO, m::DiscretePNModel) = print(io, "DiscretePNModel [$(length(dimensions(m)))D, N=$(direction_model(m).N), cells:$(n_basis(m).nx.m)]")
Base.show(io::IO, ::MIME"text/plain", m::DiscretePNModel) = show(io, m)

# function MKLSparse.SparseMatrixCSR{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
#     # very inefficient, but only do this once!
#     AT = sparse(transpose(A))
#     return MKLSparse.SparseMatrixCSR{T, Ti}(AT.m, AT.n, Vector{Ti}(AT.colptr), Vector{Ti}(AT.rowval), Vector{T}(AT.nzval))
# end

function space_model(model::DiscretePNModel)
    return model.space_mdl
end

function energy_model(model::DiscretePNModel)
    return model.energy_mdl
end

# dirac basis evaluation
function energy_eval_basis(energy_model, ϵ::Real)
    if (ϵ > energy_model[end] || ϵ < energy_model[1]) throw(ErrorException("$ϵ is not a part of $energy_model")) end
    basis = zeros(length(energy_model))
    i = findfirst(ϵi -> ϵ < ϵi, energy_model)
    if isnothing(i) # ϵi == energy_model[end]
        basis[end] = 1.0
    else
        basis[i-1] = (energy_model[i] - ϵ) / (energy_model[i] - energy_model[i-1])
        basis[i] = (ϵ - energy_model[i-1]) / (energy_model[i] - energy_model[i-1])
    end
    return basis
end

function energy_eval_basis(energy_model, f::Function)
    return f.(energy_model)
end

function direction_model(model::DiscretePNModel)
    return model.direction_mdl
end

dimensionality(model::DiscretePNModel) = SpaceModels.dimensionality(space_model(model))
dimensions(model::DiscretePNModel) = Dimensions.dimensions(dimensionality(model))

# max_degree(model::PNGridapModel) = model.direction_model
# space_directions(model::PNGridapModel) = space_directions(space_model(model))

# function function_spaces(space_model::DiscreteModel)
#     V = MultiFieldFESpace([TestFESpace(space_model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1), TestFESpace(space_model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)])
#     U = MultiFieldFESpace([TrialFESpace(V[1]), TrialFESpace(V[2])])
#     return U, V
# end

# function gridap_model(space_model::DiscreteModel)
#     U, V = function_spaces(space_model)
#     R = Triangulation(space_model)
#     ∂R = BoundaryTriangulation(space_model)
#     return U, V, (model=space_model, R=R, dx=Measure(R, 2), ∂R=∂R, dΓ= Measure(∂R, 2), n=get_normal_vector(∂R))
# end

function n_basis(model::DiscretePNModel)
    return model.number_of_basis_functions
end

function pview(v::AbstractVector, model::DiscretePNModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    return reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
end

function mview(v::AbstractVector, model::DiscretePNModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    return reshape(@view(v[nxp*nΩp+1:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
end

function pmview(v::AbstractVector, model::DiscretePNModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(model)
    @assert length(v) == nxp*nΩp + nxm*nΩm
    pview = reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
    mview = reshape(@view(v[nxp*nΩp+1:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
    return pview, mview
end

# @concrete struct MonoChromPNGridapModel{PNA<:PNArchitecture, S<:DiscreteModel} <: AbstractPNGridapModel{PNA}
#     architecture::PNA
#     space_model::S
#     direction_model
#     n_basis::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
# end

# architecture(discrete_model::MonoChromPNGridapModel) = discrete_model.architecture
# function MonoChromPNGridapModel(space_domain, direction_model, architecture=PNCPU{Float64}())
#     space_model = CartesianDiscreteModel(space_domain[1], space_domain[2])

#     U, _ = function_spaces(space_model)
#     evens = SphericalHarmonicsMatrices.get_even_moments(direction_model, nd(space_model))
#     odds = SphericalHarmonicsMatrices.get_odd_moments(direction_model, nd(space_model))

#     n_basis = ((num_free_dofs(U[1]), num_free_dofs(U[2])),
#         (length(evens), length(odds)))

#     return MonoChromPNGridapModel(
#         architecture,
#         space_model,
#         direction_model,
#         n_basis
#     )
# end

# function space(model::MonoChromPNGridapModel)
#     return model.space_model
# end

# nd(model::MonoChromPNGridapModel) = nd(space(model))

# max_degree(model::MonoChromPNGridapModel) = model.direction_model
# space_directions(model::MonoChromPNGridapModel) = space_directions(space(model))

# function number_of_basis_functions(model::MonoChromPNGridapModel)
#     nb = (x=(p=model.n_basis[1][1], m=model.n_basis[1][2]),
#           Ω=(p=model.n_basis[2][1], m=model.n_basis[2][2]))
#     return nb
# end
