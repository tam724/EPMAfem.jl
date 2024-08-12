abstract type PNArchitecture{T} end
struct PNCPU{T} <: PNArchitecture{T} end
struct PNCUDA{T} <: PNArchitecture{T} end

cpu(T=Float64) = PNCPU{T}()
cuda(T=Float32) = PNCUDA{T}()

@concrete struct PNGridapModel{PNA<:PNArchitecture, S<:DiscreteModel} # this is in fact a grid
    architecture::PNA
    space_model::S
    energy_model
    direction_model
    n_basis::Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}
end

function PNGridapModel(space_model, energy_model, direction_model, architecture=PNCPU{Float64}())
    U, _ = function_spaces(space_model)
    evens = SphericalHarmonicsMatrices.get_even_moments(direction_model, nd(space_model))
    odds = SphericalHarmonicsMatrices.get_odd_moments(direction_model, nd(space_model))

    n_basis = (length(energy_model),
        (num_free_dofs(U[1]), num_free_dofs(U[2])),
        (length(evens), length(odds)))

    return PNGridapModel(
        architecture,
        space_model,
        energy_model,
        direction_model,
        n_basis
    )
end

mat_type(::PNGridapModel{PNCPU{T}}) where {T} = Matrix{T}
smat_type(::PNGridapModel{PNCPU{T}}) where {T} = SparseMatrixCSC{T, Int64}
vec_type(::PNGridapModel{PNCPU{T}}) where {T} = Vector{T}
base_type(::PNGridapModel{PNCPU{T}}) where T = T

mat_type(::PNGridapModel{PNCUDA{T}}) where {T} = CuMatrix{T}
smat_type(::PNGridapModel{PNCUDA{T}}) where {T} = CUSPARSE.CuSparseMatrixCSC{T, Int32}
vec_type(::PNGridapModel{PNCUDA{T}}) where {T} = CuVector{T}
base_type(::PNGridapModel{PNCUDA{T}}) where T = T

# overload for the constructor of CUSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

# function MKLSparse.SparseMatrixCSR{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
#     # very inefficient, but only do this once!
#     AT = sparse(transpose(A))
#     return MKLSparse.SparseMatrixCSR{T, Ti}(AT.m, AT.n, Vector{Ti}(AT.colptr), Vector{Ti}(AT.rowval), Vector{T}(AT.nzval))
# end

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
