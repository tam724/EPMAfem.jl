module SpaceModels

using ConcreteStructs
using Gridap
using SparseArrays
using Graphs
using LinearAlgebra

using EPMAfem.Dimensions
using EPMAfem.Sparse3Tensor

include("space_model.jl")
include("space_model_matrices.jl")
include("space_model_vectors.jl")
include("space_model_tensors.jl")

export GridapSpaceModel, even, odd
export assemble_bilinear, assemble_linear

end