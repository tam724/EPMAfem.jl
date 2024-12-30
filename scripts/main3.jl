using Revise
using EPMAfem
using Plots
using CUDA
using BenchmarkTools
using LinearAlgebra

import EPMAfem.BlockedMatrices as BM




A = [discrete_problem.Ωpm[i] |> collect |> sparse for i in 1:2]
B = [discrete_problem.∇pm[i] for i in 1:2]







kron(A[1], B[1]) + kron(A[2], B[2])




A = rand(20, 30)

A_blocked = BM.blocked_from_mat(rand(20, 30), ((1:5, 1:5), (11:20, 6:10), (6:10, 11:30)), false)
A = A_blocked |> collect

A_blocked ≈ A


A_blocked
transpose(A_blocked)

A = BM.blocked_from_mat(rand(10, 10), ((1:5, 1:5), (6:10, 6:10)))

blocked = make_block(2)
blockedT = transpose(blocked)

A = collect(blocked)
AT = collect(blockedT)

A_cu = A |> cu
B = rand(10, size(blocked, 1))
B_cu = B |> cu

C1 = zeros(10, size(blocked, 2))
C2 = zeros(10, size(blocked, 2))

C_cu = CUDA.zeros(20000, size(blocked, 2))

@time mul!(C1, B, A, 1.0, 0)
@time mul!(C2, B, blocked, true, false)
C2 ≈ C1

@time mul!(C1, B, AT, 1.0, 0)
@time mul!(C2, B, blockedT, true, false)
C2 ≈ C1


@btime mul!($C2, $B, $blocked, $true, $false)
@code_warntype mul!(C2, B, blocked, true, false)

aa = rand(10, 10)
ab = rand(10, 10)
ac = rand(10, 10)

function mul_view!(aa, ab, ac, α, β)
    mul!(@view(aa[:, 1:10]), @view(ab[:, 1:10]), ac, α, β)
end

@allocated mul!(aa, ab, ac, true, false)

@btime mul_view!($aa, $ab, $ac, $true, $false)

@benchmark mul!($C1, $B, $A)
@benchmark mul!($C2, $B, $blocked)
# blocked = (blocks = (block_1, block_2, block_3, block_4), indices = ((1:50, 1:50), (51:100, 51:100), (101:150, 101:150), (151:200, 151:200)), size=(200, 200))

using CUDA
# blocked_cu = (blocks = (block_1 |> cu, block_2 |> cu), indices = ((1:50, 1:50), (51:110, 51:110)), size=(110, 110))
blocked_cu = (blocks = cu.(blocked.blocks), indices = blocked.indices, size=blocked.size)

function to_mat(blocked)
    A = zeros(blocked.size)
    for (inds, block) in zip(blocked.indices, blocked.blocks)
        A[inds[1], inds[2]] .= block
    end
    return A
end

Base.size(blocked) = blocked.size
Base.size(blocked, i) = blocked.size[i]
num_blocks(blocked) = length(blocked.blocks)


function mul_blocked!(C, B, blocked)
    for i in 1:num_blocks(blocked)
        inds = blocked.indices[i]
        block = blocked.blocks[i]
        # (inds, block) in zip(blocked.indices, blocked.blocks)
        mul!(@view(C[:, inds[2]]), @view(B[:, inds[1]]), block)
    end
end




@benchmark CUDA.@sync mul!($C_cu, $B_cu, $A_cu)
@benchmark CUDA.@sync mul_blocked!($C_cu, $B_cu, $blocked_cu)




@code_warntype mul_blocked!(C2, B, blocked)


struct MyNumber
    x::Float64
end

Base.:*(a::MyNumber, b::MyNumber, c::MyNumber) = MyNumber(a.x * b.x * c.x)

a = MyNumber(1.0)
b = MyNumber(2.0)
c = MyNumber(3.0)

a * b * c


x = rand(10, 10)
A = rand(10, 10)
y = rand(10, 10)

@edit x*A*y