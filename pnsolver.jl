abstract type PNArchitecture{T} end
struct PNCPU{T} <: PNArchitecture{T} end
struct PNCUDA{T} <: PNArchitecture{T} end

cpu(T=Float64) = PNCPU{T}()
cuda(T=Float32) = PNCUDA{T}()

index_type(::Type{Float64}) = Int64
index_type(::Type{Float32}) = Int32
index_type(::Type{Float16}) = Int32

mat_type(::PNCPU{T}) where T = Matrix{T}
smat_type(::PNCPU{T}) where T = SparseMatrixCSC{T, index_type(T)}
vec_type(::PNCPU{T}) where T = Vector{T}
base_type(::PNCPU{T}) where T = T

mat_type(::PNCUDA{T}) where T = CuMatrix{T}
smat_type(::PNCUDA{T}) where T = CUSPARSE.CuSparseMatrixCSC{T, index_type(T)}
vec_type(::PNCUDA{T}) where T = CuVector{T}
base_type(::PNCUDA{T}) where T = T

# overload for the constructor of CUSparseMatrixCSC that additionally converts the internal types.
function CUSPARSE.CuSparseMatrixCSC{T, Ti}(A::SparseMatrixCSC) where {T, Ti}
    return CUSPARSE.CuSparseMatrixCSC{T, Ti}(
        CuVector{Ti}(A.colptr), CuVector{Ti}(A.rowval), CuVector{T}(A.nzval), size(A)
    )
end

abstract type AbstractPNGridapModel{PNA<:PNArchitecture} end

mat_type(discrete_model::AbstractPNGridapModel) = mat_type(architecture(discrete_model))
smat_type(discrete_model::AbstractPNGridapModel) = smat_type(architecture(discrete_model))
vec_type(discrete_model::AbstractPNGridapModel) = vec_type(architecture(discrete_model))
base_type(discrete_model::AbstractPNGridapModel) = base_type(architecture(discrete_model))

@concrete struct PNGridapModel{PNA<:PNArchitecture, S<:DiscreteModel} <: AbstractPNGridapModel{PNA}
    architecture::PNA
    space_model::S
    energy_model
    direction_model
    n_basis::Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}
end

architecture(discrete_model::PNGridapModel) = discrete_model.architecture

function PNGridapModel(space_domain, energy_domain, direction_model, architecture::PNArchitecture{T}=PNCPU{Float64}()) where T
    space_model = CartesianDiscreteModel(space_domain[1], space_domain[2])
    U, _ = function_spaces(space_model)
    evens = SphericalHarmonicsMatrices.get_even_moments(direction_model, nd(space_model))
    odds = SphericalHarmonicsMatrices.get_odd_moments(direction_model, nd(space_model))

    energy_model = range(T.(energy_domain[1])..., energy_domain[2])
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

function gridap_model(space_model::DiscreteModel)
    U, V = function_spaces(space_model)
    R = Triangulation(space_model)
    ∂R = BoundaryTriangulation(space_model)
    return U, V, (model=space_model, R=R, dx=Measure(R, 2), ∂R=∂R, dΓ= Measure(∂R, 2), n=get_normal_vector(∂R))
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

@concrete struct MonoChromPNGridapModel{PNA<:PNArchitecture, S<:DiscreteModel} <: AbstractPNGridapModel{PNA}
    architecture::PNA
    space_model::S
    direction_model
    n_basis::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
end

architecture(discrete_model::MonoChromPNGridapModel) = discrete_model.architecture
function MonoChromPNGridapModel(space_domain, direction_model, architecture=PNCPU{Float64}())
    space_model = CartesianDiscreteModel(space_domain[1], space_domain[2])

    U, _ = function_spaces(space_model)
    evens = SphericalHarmonicsMatrices.get_even_moments(direction_model, nd(space_model))
    odds = SphericalHarmonicsMatrices.get_odd_moments(direction_model, nd(space_model))

    n_basis = ((num_free_dofs(U[1]), num_free_dofs(U[2])),
        (length(evens), length(odds)))

    return MonoChromPNGridapModel(
        architecture,
        space_model,
        direction_model,
        n_basis
    )
end

function space(model::MonoChromPNGridapModel)
    return model.space_model
end

nd(model::MonoChromPNGridapModel) = nd(space(model))

max_degree(model::MonoChromPNGridapModel) = model.direction_model
space_directions(model::MonoChromPNGridapModel) = space_directions(space(model))

function number_of_basis_functions(model::MonoChromPNGridapModel)
    nb = (x=(p=model.n_basis[1][1], m=model.n_basis[1][2]),
          Ω=(p=model.n_basis[2][1], m=model.n_basis[2][2]))
    return nb
end

abstract type PNSolver{T} end
