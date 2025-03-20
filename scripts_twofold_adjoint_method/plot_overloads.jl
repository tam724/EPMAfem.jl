using Plots
using Gridap
include("../src/space_dimensions.jl")
# Plots overloads (for fast plotting of FE functions)
import Plots: plot, plot!, heatmap, surface, contourf, contour, contour!, contourf!

function plot(x, u::CellField; kw...)
    points = Gridap.Point.(x)
    eval = u(points)
    return plot(x, eval; kw...)
end
function plot!(x, u::CellField; kw...)
    points = Gridap.Point.(x)
    eval = u(points)
    return plot!(x, eval; kw...)
end

for plot_func_2d in (:heatmap, :surface, :contourf, :contour, :contour!, :contourf!)
    @eval begin 
        function $(plot_func_2d)(x, y, u::CellField; swapxy=false, kw...)
            points = Gridap.Point.(x', y)
            eval = reshape(u(reshape(points, :)), (length(y), length(x)))
            if swapxy
                return $(plot_func_2d)(y, x, transpose(eval); kw...)
            else
                return $(plot_func_2d)(x, y, eval; kw...)
            end
        end
    end
end
