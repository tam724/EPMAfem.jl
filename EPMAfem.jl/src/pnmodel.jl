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
vec_type(Tv, ::PNCPU{T}) where T = Vector{Tv}
base_type(::PNCPU{T}) where T = T

mat_type(::PNCUDA{T}) where T = CuMatrix{T}
smat_type(::PNCUDA{T}) where T = CUSPARSE.CuSparseMatrixCSC{T, index_type(T)}
vec_type(::PNCUDA{T}) where T = CuVector{T}
vec_type(Tv, ::PNCUDA{T}) where T = CuVector{Tv}
base_type(::PNCUDA{T}) where T = T

convert_to_architecture(arch::PNArchitecture{T}, x::Matrix) where T = mat_type(arch)(x)
convert_to_architecture(arch::PNArchitecture{T}, x::SparseMatrixCSC) where T = smat_type(arch)(x)
convert_to_architecture(arch::PNArchitecture{T}, x::Vector{<:Number}) where T = vec_type(arch)(x)
convert_to_architecture(Tv, arch::PNArchitecture{T}, x::Vector{<:Number}) where T = vec_type(Tv, arch)(x)
convert_to_architecture(arch::PNArchitecture{T}, x::Vector) where T = [convert_to_architecture(arch, xi) for xi in x]
function convert_to_architecture(arch::PNArchitecture{T}, x::Sparse3Tensor.Sparse3TensorSSM) where T
    return Sparse3Tensor.Sparse3TensorSSM(
        convert_to_architecture(arch, x.skeleton),
        convert_to_architecture(arch, x.projector),
        x.size
    )
end
convert_to_architecture(arch::PNArchitecture{T}, x::Diagonal) where T = Diagonal(convert_to_architecture(arch, x.diag))

allocate_vec(arch::PNArchitecture{T}, n::Int) where T = vec_type(arch)(undef, n)
allocate_mat(arch::PNArchitecture{T}, m::Int, n::Int) where T = mat_type(arch)(undef, m, n)

abstract type AbstractPNGridapModel{PNA<:PNArchitecture} end

@concrete struct PNGridapModel{PNA<:PNArchitecture} <: AbstractPNGridapModel{PNA}
    architecture::PNA
    space_model
    energy_model
    direction_model
    n_basis::Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}
end

architecture(discrete_model::PNGridapModel) = discrete_model.architecture

function PNGridapModel(space_model, energy_model, direction_model, architecture::PNArchitecture{T}=PNCPU{Float64}()) where T
    @assert SpaceModels.dimensionality(space_model) == SphericalHarmonicsModels.dimensionality(direction_model)
    n_basis_space = SpaceModels.n_basis(space_model)
    n_basis_direction = SphericalHarmonicsModels.n_basis(direction_model)
    n_basis_energy = length(energy_model)

    n_basis = (n_basis_energy,
        (n_basis_space.p, n_basis_space.m),
        (n_basis_direction.p, n_basis_direction.m))

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

function direction(model::PNGridapModel)
    return model.direction_model
end

dimensionality(model::PNGridapModel) = SpaceModels.dimensionality(space(model))
dimensions(model::PNGridapModel) = Dimensions.dimensions(dimensionality(model))

# max_degree(model::PNGridapModel) = model.direction_model
# space_directions(model::PNGridapModel) = space_directions(space(model))

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

function number_of_basis_functions(model::PNGridapModel)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = model.n_basis
    x = (p=nxp, m=nxm)
    Ω = (p=nΩp, m=nΩm)
    return (x=x, Ω=Ω)
end

@inline function pview(v::AbstractVector, model::PNGridapModel)
    (_, (nxp, _), (nΩp, _)) = model.n_basis
    return reshape(@view(v[1:nxp*nΩp]), (nxp, nΩp))
end

@inline function mview(v::AbstractVector, model::PNGridapModel)
    (_, (nxp, nxm), (nΩp, nΩm)) = model.n_basis
    return reshape(@view(v[nxp*nΩp+1:nxp*nΩp + nxm*nΩm]), (nxm, nΩm))
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
