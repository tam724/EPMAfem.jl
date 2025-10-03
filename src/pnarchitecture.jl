abstract type PNArchitecture{T} end
struct PNCPU{T} <: PNArchitecture{T} end
struct PNCUDA{T} <: PNArchitecture{T} end

cpu(T=Float64) = PNCPU{T}()
cuda(T=Float32) = PNCUDA{T}()

mat_type(::PNCPU{T}) where T = Matrix{T}
vec_type(::PNCPU{T}) where T = Vector{T}
vec_type(Tv, ::PNCPU{T}) where T = Vector{Tv}
arr_base_type(::PNCPU{T}) where T = Array
arr_type(::PNCPU{T}) where T = Array{T}
base_type(::PNCPU{T}) where T = T

mat_type(::PNCUDA{T}) where T = CuMatrix{T}
vec_type(::PNCUDA{T}) where T = CuVector{T}
vec_type(Tv, ::PNCUDA{T}) where T = CuVector{Tv}
arr_type(::PNCUDA{T}) where T = CuArray{T}
arr_base_type(::PNCUDA{T}) where T = CuArray
base_type(::PNCUDA{T}) where T = T

allocate_vec(arch::PNArchitecture{T}, n::Int) where T = arr_type(arch)(undef, n)
allocate_mat(arch::PNArchitecture{T}, m::Int, n::Int) where T = arr_type(arch)(undef, m, n)
allocate_arr(arch::PNArchitecture{T}, m::Int, n::Int, k::Int) where T = arr_type(arch)(undef, m, n, k)

## syntactic sugar

(arch::PNArchitecture)(x) = Adapt.adapt(arr_type(arch), x)
(arch::PNArchitecture)(Tv, x) = Adapt.adapt(arr_base_type(arch){Tv}, x)
