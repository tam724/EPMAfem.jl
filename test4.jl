using LinearAlgebra

struct TestStruct
    a::Vector{Float64}, b::Vector{Float64}, c::Vector{Float64}
end



test = TestStruct(rand(10), rand(10), rand(10))
function print((; a, b, c)::TestStruct)
    @show a
end

print(test)

import LinearAlgebra: mul!
function LinearAlgebra.mul!(c::AbstractVector{T}, (; L, R, tmp)::@NamedTuple{L::M, R::M, tmp::V}, b::AbstractVector{T}) where {T, M<:AbstractMatrix{T}, V<:AbstractVector{T}}
    Tmp = reshape(@view(tmp[:]), (10, 10))
    B = reshape(@view(b[:]), (10, 10))
    C = reshape(@view(c[:]), (10, 10))
    mul!(Tmp, B, R, true, false)
    mul!(C, L, Tmp)
    return 
end

function LinearAlgebra.mul!(c::AbstractVector{T}, (; Ls, Rs, tmp)::@NamedTuple{Ls::M, Rs::M2, tmp::V}, b::AbstractVector{T}) where {T, M, M2, V}
    Tmp = reshape(@view(tmp[:]), (10, 10))
    # @show size(Tmp)
    B = reshape(@view(b[:]), (10, 10))
    C = reshape(@view(c[:]), (10, 10))
    for (L, R) in zip(Ls, Rs)
        mul!(Tmp, B, R, true, false)
        mul!(C, L, Tmp)
    end
    return 
end


kronmatrix = (L=rand(10, 10), R=rand(10, 10), tmp=zeros(10*10))
kronsmatrix = (Ls=[rand(10, 10) for _ in 1:3], Rs=[rand(10, 10) for _ in 1:3], tmp=zeros(10*10))

C = rand(100000)
B = rand(100)

using BenchmarkTools
@benchmark mul!(C, kronmatrix, B)
@benchmark mul!($C, $kronsmatrix, $B)
@code_warntype mul!(C, kronsmatrix, B)

mul!(rand(10, 10), rand(10, 10), rand(10, 10))

function my_rmul!(A, b)
    if b return
    else rmul!(A, b)
    end
end

using CUDA

C_cu = cu(C)

@which rmul!(C_cu, true)
@benchmark my_rmul!(C, true)

test!(C, kronmatrix, B)

if 1.0
    @show "test"
end

test = Diagonal(rand(10))
test2 = rand(10, 10) #(rand(10))

@time axpy!(1.0, test, test2)