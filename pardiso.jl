struct Pardiso end

using Libdl
using SparseArrays
# function solve(A, b)
#     @ccall


pardiso_path = "/home/tam/software/panua-pardiso-20240229-linux/lib/libpardiso.so"

pardiso_lib = Libdl.dlopen(pardiso_path)

A = sprand(100, 100, 0.1)
CRS

SparseMatrixCSR