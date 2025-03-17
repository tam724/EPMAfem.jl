module EPMAfemTests

using Test

@time @testset "SphericalHarmonicsModels" begin include("spherical_harmonics_test.jl") end
@time @testset "Sparse3Tensor" begin include("sparse3tensor_test.jl") end
@time @testset "BlockedMatrices" begin include("blockedmatrices_test.jl") end
@time @testset "OnlyEnergy" begin include("test_only_energy/onlyenergy_test.jl") end
@time @testset "DMatrixTest" begin include("dmatrix_test.jl") end
@time @testset "ZMatrixTest" begin include("zmatrix_test.jl") end
@time @testset "BlockSchurMatrixTest" begin include("block_schur_mat_test.jl") end

end
