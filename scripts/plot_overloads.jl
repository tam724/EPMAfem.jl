using Plots
using Gridap
include("../EPMAfem.jl/src/space_dimensions.jl")
# Plots overloads (for fast plotting of FE functions)
import Plots: plot, plot!, heatmap, surface
function plot(x, u::CellField; kw...)
    points = Point.(x)
    eval = u(points)
    return plot(x, eval; kw...)
end
function plot!(x, u::CellField; kw...)
    points = Point.(x)
    eval = u(points)
    return plot!(x, eval; kw...)
end
function heatmap(x, y, u::CellField; kw...)
    points = Point.(x', y)
    eval = reshape(u(reshape(points, :)), (length(y), length(x)))
    return heatmap(x, y, eval; kw...)
end
function surface(x, y, u::CellField; kw...)
    points = Point.(x', y)
    eval = reshape(u(reshape(points, :)), (length(y), length(x)))
    return surface(x, y, eval; kw...)
end
function sphere_surf(u::Function; kw...)
    plotly()
    θ = range(0, π, length=180)
    ϕ = range(0, 2π, length=360)
    Ωs = Dimensions.unitsphere_spherical_to_cartesian.(tuple.(θ, ϕ'))
    h = u.(Ωs)
    x, y, z = Dimensions.Ωx.(Ωs), Dimensions.Ωy.(Ωs), Dimensions.Ωz.(Ωs)
    # @show x, y, z
    surface(x, y, z; fill_z=h)
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")
end