using LinearAlgebra, Krylov, IterativeSolvers
using Plots


N = 100

B = rand(N, N);
C_ = rand(N, N)
C = C_*C_'./norm(C_)
D = Diagonal(rand(N))

A = [C B
transpose(B) D];

maximum(abs.(A .- transpose(A)))

b = rand(size(A, 2))

A\b

import LinearAlgebra: mul!
function mul!(y, A::NamedTuple, x, α, β)
    @assert β == false
    mul!(A.tmp, transpose(A.B), x, α, false)
    mul!(y, A.D_inv, A.tmp, true, false)
    mul!(A.tmp, A.B, y, true, false)
    mul!(y, A.C, x, α, false)
    y .-= A.tmp
end

import Base: size
function size(A::NamedTuple)
    return size(A.C)
end

function size(A::NamedTuple, i)
    return size(A.C, i)
end

function solve_schur((C, B, D), b)
	a = @view(b[1:N])
	c = @view(b[N+1:N+N])
	D_inv = inv(D)
	temp_mat = (C=C, B=B, D=D, D_inv=D_inv, tmp=zeros(N))
	# x, stats = Krylov.gmres(C - B*D_inv*transpose(B), a - B*D_inv*c)
	
	A_schur = C - B*D_inv*transpose(B)
	# @show maximum(abs.(A_schur .- transpose(A_schur)))
	
	#x, stats = Krylov.minres(temp_mat, a - B*D_inv*c, atol=1e-20, rtol=1e-20)
	x, stats = Krylov.gmres(temp_mat, a - B*D_inv*c)

	@show maximum(abs.((A_schur*x) .- (a - B*D_inv*c)))

	y = D_inv*(c - transpose(B)*x)
	return [x
	y], stats
end

@time x_schur, stats = solve_schur((C, B, D), b)

@time x_gmres, _ = Krylov.gmres(A, b)

eigvals(A)