using Revise

using EPMAfem
using EPMAfem.CUDA
include("plot_overloads.jl")

using Zygote
#using GLMakie

#import EPMAfem.SphericalHarmonicsModels as SH
#import EPMAfem.SpaceModels as SM
using LinearAlgebra
using Gridap
using GLMakie
#using StaticArrays

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5, -0.5, 0.5), (3, 3)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(21, 3)
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(3, 3)
SH = EPMAfem.SphericalHarmonicsModels

for i in 1:length(SH.even(direction_model))
    p = sphere_surf(Ω -> SH.even(direction_model)[i](Ω), colormap=:temperaturemap, axis=(show_axis=false,), figure=(size=(800, 800), ))
    save("basis_functions/spherical_harmonics/even_$i.png", p)
end

for i in 1:length(SH.odd(direction_model))
    p = sphere_surf(Ω -> SH.odd(direction_model)[i](Ω), colormap=:temperaturemap, axis=(show_axis=false,), figure=(size=(800, 800), ))
    save("basis_functions/spherical_harmonics/odd_$i.png", p)
end


SM = EPMAfem.SpaceModels

x = -0.5:0.01:0.5
y = -0.5:0.01:0.5

xy = Gridap.Point.(x, y')
for i in 1:num_free_dofs(SM.even(space_model))    
    e_i = zeros(num_free_dofs(SM.even(space_model)))
    e_i[i] = 1.0
    func = FEFunction(SM.even(space_model), e_i)

    p = Makie.surface(x, y, func.(xy), colormap=:temperaturemap, axis=(show_axis=false,), figure=(size=(800, 800), ))
    save("basis_functions/finite_elements/even_$(i).png", p)
end

for i in 1:num_free_dofs(SM.odd(space_model))    
    e_i = zeros(num_free_dofs(SM.odd(space_model)))
    e_i[i] = 1.0
    func = FEFunction(SM.odd(space_model), e_i)

    p = Makie.surface(x, y, func.(xy), colormap=:temperaturemap, axis=(show_axis=false,), figure=(size=(800, 800), ))
    save("basis_functions/finite_elements/odd_$(i).png", p)
end