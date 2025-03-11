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

function sphere_surf(u::Function; kw...)
    # plotlyjs()
    θ = range(0.0001, π-0.0001, length=180)
    ϕ = range(0, 2π, length=360)
    Ωs = Dimensions.unitsphere_spherical_to_cartesian.(tuple.(θ, ϕ'))
    h = u.(Ωs)
    x, y, z = Dimensions.Ωx.(Ωs), Dimensions.Ωy.(Ωs), Dimensions.Ωz.(Ωs)
    # @show x, y, z
    # surface(x, y, z; fill_z=h, kw...)
    Makie.surface(x, y, z; color=h, kw...)
    # xlabel!("x")
    # ylabel!("y")
    # zlabel!("z")
end

function circle_quiver(u::Function, o_z=0, o_x=0; kw...)
    θ = range(0, 2π, length=90)
    Ωs = VectorValue.(sin.(θ), cos.(θ))
    z, x = Dimensions.Ωz.(Ωs), Dimensions.Ωx.(Ωs)
    h = u.(Ωs)

    x_, y_, u_, v_ = zeros(length(θ)), zeros(length(θ)), zeros(length(θ)), zeros(length(θ))
    for i in 1:length(θ)
        if h[i] > 0
            #normal quiver
            x_[i], y_[i] = o_x, o_z
            u_[i], v_[i] = h[i].*(x[i], z[i])
        end
    end
    Plots.quiver(x_, y_, quiver=(u_, v_))
    Plots.xlims!(o_x-maximum(abs.(h)), o_x+maximum(abs.(h)))
    Plots.ylims!(o_z-maximum(abs.(h)), o_z+maximum(abs.(h)))
    Plots.xlabel!("x")
    Plots.ylabel!("z")
end

for (func_name, plot_func_name) in [(:circle_lines, :(Plots.plot)), (:circle_lines!, :(Plots.plot!))]
@eval begin function $(func_name)(u::Function, o_z=0, o_x=0, scale=1; kw...)
            θ = range(0, 2π, length=360)
            Ωs = VectorValue.(sin.(θ), cos.(θ))
            z, x = Dimensions.Ωz.(Ωs), Dimensions.Ωx.(Ωs)
            h = u.(Ωs)
            #h[h.<0] .= 0.0
            h = normalize(h)

            x_, y_ = zeros(length(θ)+1), zeros(length(θ)+1)
            for i in 1:length(θ)
                if h[i] > 0
                    x_[i], y_[i] = o_x + scale*h[i]*x[i], o_z + scale*h[i]*z[i]
                    # u_[i], v_[i] = h[i].*(x[i], z[i])
                else
                    x_[i], y_[i] = o_x, o_z
                end
            end
            x_[end], y_[end] = o_x + scale*h[1]*x[1], o_z + scale*h[1]*z[1]
            $(plot_func_name)(x_, y_; kw...)

            x_, y_ = zeros(length(θ)+1), zeros(length(θ)+1)
            for i in 1:length(θ)
                if h[i] < 0
                    x_[i], y_[i] = o_x - scale*h[i]*x[i], o_z - scale*h[i]*z[i]
                    # u_[i], v_[i] = h[i].*(x[i], z[i])
                else
                    x_[i], y_[i] = o_x, o_z
                end
            end
            x_[end], y_[end] = o_x - scale*h[1]*x[1], o_z - scale*h[1]*z[1]
            $(plot_func_name)(x_, y_, label=nothing; color=:red)
            # Plots.xlims!(o_x-scale*maximum(abs.(h)), o_x+scale*maximum(abs.(h)))
            # Plots.ylims!(o_z-scale*maximum(abs.(h)), o_z+scale*maximum(abs.(h)))
            # Plots.xlabel!("x")
            # Plots.ylabel!("z")
        end
    end
end