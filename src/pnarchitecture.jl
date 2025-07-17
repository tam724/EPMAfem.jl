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
svec_type(::PNCPU{T}) where T = SparseVector{T, index_type(T)}
vec_type(Tv, ::PNCPU{T}) where T = Vector{Tv}
arr_type(::PNCPU{T}) where T = Array{T}
base_type(::PNCPU{T}) where T = T

mat_type(::PNCUDA{T}) where T = CuMatrix{T}
smat_type(::PNCUDA{T}) where T = CUSPARSE.CuSparseMatrixCSC{T, index_type(T)}
vec_type(::PNCUDA{T}) where T = CuVector{T}
svec_type(::PNCUDA{T}) where T = CUSPARSE.CuSparseVector{T, index_type(T)}
vec_type(Tv, ::PNCUDA{T}) where T = CuVector{Tv}
arr_type(::PNCUDA{T}) where T = CuArray{T}
base_type(::PNCUDA{T}) where T = T

# types that must be converted to the architecture
convert_to_architecture(arch::PNArchitecture{T}, x::Matrix) where T = if x isa mat_type(arch) return x else return mat_type(arch)(x) end
convert_to_architecture(arch::PNArchitecture{T}, x::SparseMatrixCSC) where T = if x isa smat_type(arch) return x else return smat_type(arch)(x) end
convert_to_architecture(arch::PNArchitecture{T}, x::Vector{<:Number}) where T = if x isa vec_type(arch) return x else return vec_type(arch)(x) end
convert_to_architecture(arch::PNArchitecture{T}, x::SparseVector{<:Number}) where T = if x isa svec_type(arch) return x else return svec_type(arch)(x) end
convert_to_architecture(Tv, arch::PNArchitecture{T}, x::Vector{<:Number}) where T = if x isa vec_type(Tv, arch) return x else return vec_type(Tv, arch)(x) end
convert_to_architecture(arch::PNArchitecture{T}, x::AbstractArray{<:Number}) where T = if x isa arr_type(arch) return x else return arr_type(arch)(x) end

# wrapper types (we convert the inner data)
convert_to_architecture(arch::PNArchitecture{T}, x::Union{Vector, NTuple}) where T = convert_to_architecture.(Ref(arch), x)
# we support one named tuple
convert_to_architecture(arch::PNArchitecture{T}, x::@NamedTuple{p::AT, m::AT}) where {T, AT} = (p=convert_to_architecture(arch, x.p), m=convert_to_architecture(arch, x.m))
convert_to_architecture(arch::PNArchitecture{T}, x::BlockedMatrices.BlockedMatrix) where T = BlockedMatrices.BlockedMatrix(convert_to_architecture(arch, x.blocks), x.indices, x.axes)
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
allocate_arr(arch::PNArchitecture{T}, m::Int, n::Int, k::Int) where T = arr_type(arch)(undef, m, n, k)

## syntactic sugar
(arch::PNArchitecture)(x) = convert_to_architecture(arch, x)
(arch::PNArchitecture)(Tv, x) = convert_to_architecture(Tv, arch, x)
