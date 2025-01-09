using Plots
using Gridap
include("../EPMAfem.jl/src/space_dimensions.jl")
# Plots overloads (for fast plotting of FE functions)
import Plots: plot, plot!, heatmap, surface, contourf, contour

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

for plot_func_2d in (:heatmap, :surface, :contourf, :contour)
    @eval begin 
        function $(plot_func_2d)(x, y, u::CellField; swapxy=false, kw...)
            points = Point.(x', y)
            eval = reshape(u(reshape(points, :)), (length(y), length(x)))
            if swapxy
                return $(plot_func_2d)(y, x, transpose(eval); kw...)
            else
                return $(plot_func_2d)(x, y, eval; kw...)
            end
        end
    end
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