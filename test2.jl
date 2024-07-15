using IterativeSolvers
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Krylov
using CUDA

struct A{L_, R_, temp_}
    L::L_
    R::R_
    temp::temp_
end

import LinearAlgebra: mul!
import Base: size, eltype
import CUDA: cu

function mul!(b::AbstractVector, A_::A{L, R, T}, x::AbstractVector) where {L, R, T}
    n, m = size(A_.L)
    o, p = size(A_.R)
    @assert(size(x, 1) == size(A_, 2))
    @assert(size(b, 1) == size(A_, 1))
    X1 = reshape(@view(x[1:o*m]), (m, o))
    X2 = reshape(@view(x[o*m+1:end]), (m, o))
    B1 = reshape(@view(b[1:p*n]), (n, p))
    B2 = reshape(@view(b[p*n+1:end]), (n, p))
    mul!(A_.temp, X1, A_.R)
    mul!(B1, A_.L, A_.temp)
    mul!(A_.temp, X2, A_.R)
    mul!(B2, A_.L, A_.temp)
    return b
end

function to_mat(A_::A)
    A_mat = zeros(size(A_))
    temp = zeros(size(A_, 1))
    e_i = zeros(size(A_, 2))
    for i in 1:size(A_, 2)
        e_i[i] = 1.0
        # @show typeof.([temp, A_, e_i])
        mul!(temp, A_, e_i)
        A_mat[:, i] = temp
        e_i[i] = 0.0
    end
    return A_mat
end 

function eltype(A_::A)
    return eltype(A_.L)
end

function size(A_::A)
    return (size(A_, 1), size(A_, 2))
end

function size(A_::A, i)
    n, m = size(A_.L)
    o, p = size(A_.R)

    if i == 1
        return p*n*2
    elseif i == 2
        return o*m*2
    else
        @assert(false)
    end
end

function cu(A_::A)
    return A(cu(A_.L), cu(A_.R), cu(A_.temp))
end

n = 100
L = Matrix(1.0*I, (n, n)) .+ 0.1*rand(n, n)
R = Matrix(1.0*I, (n, n))

A_default = [kron(transpose(R), L) zeros(n*n, n*n)
    zeros(n*n, n*n) kron(transpose(R), L)]

A_new = A(L, R, zeros(n, n))
#A_test = to_mat(A_new)

maximum(abs.(A_test .- A_default))

A_cu = cu(A_new)

x = rand(2*n*n)
b = zeros(2*n*n)
b_default = zeros(2*n*n)

x_cu = cu(x)
b_cu = cu(b)
@benchmark mul!(b_cu, A_cu, x_cu)
@benchmark mul!(b, A_new, x)
@benchmark mul!(b_default, A_default, x)

maximum(abs.(b .- b_default))
maximum(abs.(Vector(b_cu) .- b_default))

x = zeros(2*n*n)
x_default = zeros(2*n*n)
b = rand(2*n*n)
x_cu = cu(x)
b_cu = cu(b)

@benchmark IterativeSolvers.bicgstabl!(x_default, A_default, b, log=true)
@benchmark IterativeSolvers.bicgstabl!(x, A_new, b, log=true)
@benchmark IterativeSolvers.bicgstabl!(x_cu, A_cu, b_cu, log=true)

maximum(abs.(x_default .- x))
maximum(abs.(x_default .- A_default\b))
maximum(abs.(x .- A_default\b))

@benchmark x_new = Krylov.bicgstab(A_new, b, atol=1e-15, rtol=1e-15, itmax=100)[1]
@benchmark x_default = Krylov.bicgstab(A_default, b, atol=1e-15, rtol=1e-15, itmax=100)[1]
@benchmark x_cu = Krylov.bicgstab(A_cu, b_cu, atol=Float32(1e-15), rtol=Float32(1e-15), itmax=100)[1]

using Plots

x_new .- x_default