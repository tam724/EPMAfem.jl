using Revise
using EPMAfem
using LinearAlgebra
using SparseArrays

function rand_tensor((n1, n2, n3), nvals)
    I = rand(1:n1, nvals)
    J = rand(1:n2, nvals)
    K = rand(1:n3, nvals)

    V = rand(nvals)

    return EPMAfem.Sparse3Tensor.sparse3tensor(I, J, K, V, (n1, n2, n3))
end

tens = rand_tensor((5, 5, 5), 10)
tensor = EPMAfem.Sparse3Tensor.convert_to_SSM(tens)

u = rand(5, 10)
v = rand(5, 10)

using Zygote

function tensordot2(A::EPMAfem.Sparse3Tensor.Sparse3TensorSSM, u, v, w::AbstractVector{T}) where T
    β = false
	for pr_ in A.projector
		mul!(nonzeros(A.skeleton), pr_, w, true, β)
        β = true
	end
    return dot(u, A.skeleton, v)
end

Zygote.@adjoint function tensordot2(A::EPMAfem.Sparse3Tensor.Sparse3TensorSSM, u, v, w::AbstractVector{T}) where T
    β = false
	for pr_ in A.projector
		mul!(nonzeros(A.skeleton), pr_, w, true, β)
        β = true
	end
    Σ = dot(u, A.skeleton, v)
    function tensordot2_pullback(Δ)
        Δ_skeleton = similar(nonzeros(A.skeleton))
        i_nz = 0
        for j in 1:size(A.skeleton, 2)
            for rowptr in nzrange(A.skeleton, j)
                i = A.skeleton.rowval[rowptr]
                i_nz += 1
                Δ_skeleton[i_nz] = dot(@view(u[i, :]), @view(v[j, :]))
            end
        end
        Δ_w = similar(w)
        β = false
        for pr_ in A.projector
            mul!(Δ_w, transpose(pr_), Δ_skeleton, Δ, β)
            β = true
        end
        # Δ_projector = [mul!(zeros(T, length(pr)), Δ_skeleton, w, false, false) for pr in A.projector]
        return (nothing, nothing, nothing, Δ_w)
    end
    return Σ, tensordot2_pullback
end

func(w) = tensordot2(tensor, u, v, w)

w = rand(5)
func(w)

Zygote.gradient(func, w)

wΔ = zeros(5)
wΔ[2] += 1e-4
(func(w .+ wΔ) - func(w)) / 1e-4

A = tensor.skeleton


f(A) = dot(u, A, v)

grad = Zygote.gradient(f, A)