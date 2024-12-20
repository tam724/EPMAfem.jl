module EPMAfemTests

using Test

@time @testset "SphericalHarmonicsModels" begin include("spherical_harmonics_test.jl") end


end