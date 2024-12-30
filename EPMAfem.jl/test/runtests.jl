module EPMAfemTests

using Test

@time @testset "SphericalHarmonicsModels" begin include("spherical_harmonics_test.jl") end
@time @testset "Sparse3Tensor" begin include("sparse3tensor_test.jl") end
@time @testset "BlockedMatrices" begin include("blockedmatrices_test.jl") end
end