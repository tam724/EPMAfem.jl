module SpaceModels

using ConcreteStructs
using Gridap
using SparseArrays
using Graphs
using LinearAlgebra

using EPMAfem.Dimensions

include("space_model.jl")
include("space_model_matrices.jl")
include("space_model_vectors.jl")
include("trilinearform.jl")

export GridapSpaceModel, even, odd
export assemble_bilinear, assemble_linear

end