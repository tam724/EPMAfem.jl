using LinearAlgebra
BLAS.get_num_threads()
BLAS.get_config()
BLAS.set_num_threads(8)
using BenchmarkTools

N = 5000
A = rand(N, N);
B = rand(N, N);
C = zeros(N, N);

@benchmark mul!(C, A, B)

using BLISBLAS
BLISBLAS.get_num_threads()
BLAS.get_config()
@benchmark mul!(C, A, B)

using MKL
