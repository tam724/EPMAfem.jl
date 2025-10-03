using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Debugger

Lazy = EPMAfem.PNLazyMatrices


A = lazy(Diagonal(rand(2)))
B = lazy(rand(2, 2))


(A, B) isa NTuple{2, AbstractMatrix{Float64}}

C = lazy(rand(2, 2))
D = lazy(rand(2, 2))

M = EPMAfem.materialize(EPMAfem.materialize(kron(A, B)) + EPMAfem.materialize(kron(C, D)))

M = EPMAfem.materialize(A + 2.0 * C)

ws = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.materialize_with, M, ()), zeros)
EPMAfem.materialize_with(ws, M)


Lazy.materialize_broadcasted(ws, M)[1]
