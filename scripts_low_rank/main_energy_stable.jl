using Revise
using EPMAfem
using LinearAlgebra
using EPMAfem.SphericalHarmonicsModels
using EPMAfem.SpaceModels
SH = SphericalHarmonicsModels
SM = SpaceModels

using EPMAfem.Gridap


for d in (1, 2, 3)
    for N in 1:1:27
        direction_model = SH.EOSphericalHarmonicsModel(10, 2)
        A = SH.assemble_bilinear(SH.∫S²_absΩzuv, direction_model, SH.even(direction_model), SH.even(direction_model))
        @show all(eigen(A).values .> 0.0)
        @assert all(eigen(A).values .> 0.0)
        if d > 1
            A = SH.assemble_bilinear(SH.∫S²_absΩxuv, direction_model, SH.even(direction_model), SH.even(direction_model))
            @show all(eigen(A).values .> 0.0)
            @assert all(eigen(A).values .> 0.0)
            if d > 2
                A = SH.assemble_bilinear(SH.∫S²_absΩyuv, direction_model, SH.even(direction_model), SH.even(direction_model))
                @show all(eigen(A).values .> 0.0)
                @assert all(eigen(A).values .> 0.0)
            end
        end
    end
end

space_model = SM.GridapSpaceModel(CartesianDiscreteModel((0, 1), 10))
A = SM.assemble_bilinear(SM.∫∂R_absnz_uv, space_model, SM.even(space_model), SM.even(space_model))
all(eigen(A).values .>= 0)
