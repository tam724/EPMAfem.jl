using Revise
using NeXLCore
using EPMAfem
using NeXLCore.Unitful
using Plots
using Gridap

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)
dimless, dimful = NExt.dimless, NExt.dimful

mat = [n"Cu", n"Pt", n"Fe"]
k_rats = [n"Cu K-L2", n"Pt L3-M5", n"Fe K-L2"]
ϵ_range = range(5u"keV", 17.0u"keV", length=50)
eq = NExt.epma_equations(mat, ϵ_range, 13)
model = NExt.epma_model(eq, (-1200.0u"nm", 0.0u"nm", -1200.0u"nm", 1200.0u"nm", -1200.0u"nm", 1200.0u"nm"), (30, 30, 30), 13)
# model = NExt.epma_model(eq, (-1500.0u"nm", 0.0u"nm", -1500u"nm", 1500u"nm", -1500u"nm", 1500u"nm"), (2, 2, 2), 13)

upd_pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=true)
pnsystem = EPMAfem.implicit_midpoint(upd_pnproblem.problem, EPMAfem.PNSchurSolver)

# outflux = EPMAfem.discretize_outflux(model, EPMAfem.cuda())

excitation = EPMAfem.pn_excitation([(x=dimless(x_, eq.dim_basis), y=dimless(y_, eq.dim_basis)) for x_ in range(-700u"nm", 700u"nm", length=20) for y_ in range(-700u"nm", 700u"nm", length=20)], [dimless(15.0u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=dimless(50u"nm", eq.dim_basis), beam_energy_σ=dimless(0.5u"keV", eq.dim_basis), beam_direction_κ=50.0)
@profview discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())



# (should be in the library somwhere)
discrete_ext = let

    idx_k_rat_mat = 3
    T = EPMAfem.base_type(EPMAfem.cuda())

    ## instantiate Gridap
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    space_mdl = EPMAfem.space_model(model)
    direction_mdl = EPMAfem.direction_model(model)

    ## ... and extraction
    μϵ = Vector{T}([NeXLCore.ionizationcrosssection(k_rats[idx_k_rat_mat] |> NeXLCore.inner, dimful(ϵ, u"eV", eq.dim_basis) |> ustrip) for ϵ ∈ EPMAfem.energy_model(model)])
    # normalize 
    μϵ .= μϵ ./ maximum(μϵ)

    μΩp = SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_mdl, SH.even(direction_mdl)) |> EPMAfem.cuda()

    ρ_proj = SM.assemble_bilinear(SM.∫R_uv, space_mdl, SM.odd(space_mdl), SM.even(space_mdl))
    # ρs = EPMAfem.discretize_mass_concentrations(eq, model)
    n_parameters = (EPMAfem.number_of_elements(eq), EPMAfem.n_basis(model).nx.m)
    EPMAfem.UpdatableRank1DiscretePNVector(EPMAfem.Rank1DiscretePNVector(true, model, EPMAfem.cuda(), μϵ, ρ_proj*@view(ρs[idx_k_rat_mat, :]) |> EPMAfem.cuda(), μΩp), ρ_proj, n_parameters, idx_k_rat_mat)
end

function is_in_core(x, y)
    x_center = (0u"nm", -600u"nm", -600u"nm", 600u"nm", 600u"nm")
    y_center = (0u"nm", -600u"nm", 600u"nm", -600u"nm", 600u"nm")
    return any(((xc, yc), ) -> sqrt((x-xc)^2 + (y-yc)^2) <= 170u"nm", zip(x_center, y_center))
end

function is_in_mantle(x, y)
    x_center = (0u"nm", -600u"nm", -600u"nm", 600u"nm", 600u"nm")
    y_center = (0u"nm", -600u"nm", 600u"nm", -600u"nm", 600u"nm")
    return any(((xc, yc), ) -> sqrt((x-xc)^2 + (y-yc)^2) > 170u"nm" && sqrt((x-xc)^2 + (y-yc)^2) < 290u"nm", zip(x_center, y_center))
end 

function mass_concentrations(e, x_, to_nm)
    z = x_[1]*to_nm
    x = x_[2]*to_nm
    y = x_[3]*to_nm
    if is_in_core(x, y)
        return e == 2 ? dimless(n"Pt".density, eq.dim_basis) : 0.0
    elseif is_in_mantle(x, y)
        return e == 3 ? dimless(n"Fe".density, eq.dim_basis) : 0.0
    else
        return e == 1 ? dimless(n"Cu".density, eq.dim_basis) : 0.0
    end

    # if sqrt((x-700u"nm")^2 + (y-700u"nm")^2 + (z-100u"nm")^2) < 400u"nm"
    #     return e == 2 ? dimless(mat[2].density, eq.dim_basis) : 0.0
    # elseif sqrt(x^2 + (y+700u"nm")^2) < 500u"nm"
    #     return e == 2 ? dimless(mat[2].density, eq.dim_basis) : 0.0
    # else
    #     return e == 1 ? dimless(mat[1].density, eq.dim_basis) : 0.0
    # end
end

to_nm = fac_dimful(u"nm", eq.dim_basis)*u"nm"

heatmap(range(-700.0u"nm", 700u"nm", 70), range(-700.0u"nm", 700u"nm", 70), (x, y) -> mass_concentrations(3, VectorValue(0.0, dimless(x, eq.dim_basis), dimless(y, eq.dim_basis)), to_nm))

ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x, to_nm) for e in 1:length(mat)], model)

EPMAfem.update_problem!(upd_pnproblem, ρs)
EPMAfem.update_vector!(discrete_ext, ρs)
# plot(range(-1.0, 1.0, length=100), x -> EPMAfem.beam_space_distribution(ex, 50, (0.0, x, 0.0)))
# plot(range(-500u"nm", 500u"nm", length=100), x -> EPMAfem.beam_space_distribution(ex, 50, (0.0, dimless(x, eq.dim_basis), 0.0)))
# plot!(-500:10:500, x -> pdf(Normal(0.0, 50.0), x) / pdf(Normal(0.0, 50.0), 0))

# plot(ϵ_range, ϵ -> EPMAfem.beam_energy_distribution(ex, 1, dimless(ϵ, eq.dim_basis)))

# x = getindex.(model.space_mdl.discrete_model.grid.node_coords, 1)
# y = getindex.(model.space_mdl.discrete_model.grid.node_coords, 2)
# scatter(x, y)

#plot(inflx[1, :, 1])

@time res = (discrete_ext.vector * pnsystem) * discrete_rhs

using Serialization

serialize("res_Fe.jls", res)
heatmap(range(-700u"nm", 700u"nm", 70), range(-700u"nm", 700u"nm", 70), reshape(res, (70, 70)), colorbar=true, aspect_ratio=:equal)



## plotting

res_Pt = reshape(deserialize("res_Pt.jls"), (70, 70))
res_Cu = reshape(deserialize("res_Cu.jls"), (70, 70))
res_Fe = reshape(deserialize("res_Fe.jls"), (70, 70))

res_Pt = res_Pt ./ maximum(res_Pt)
res_Fe = res_Fe ./ maximum(res_Fe)

heatmap(RGB.(res_Pt, res_Fe, zeros(size(res_Pt))))

heatmap(range(-700u"nm", 700u"nm", 70), range(-700u"nm", 700u"nm", 70), res_Fe, color=cgrad([:black, :red]), colorbar=false, aspect_ratio=:equal)
plot!(size=(300, 285), dpi=800, fontfamily="Computer Modern")
savefig("k_ratios_Fe.png")

heatmap(range(-700u"nm", 700u"nm", 70), range(-700u"nm", 700u"nm", 70), res_Pt, color=cgrad([:black, :lightgreen]), colorbar=false, aspect_ratio=:equal)
plot!(size=(300, 285), dpi=800, fontfamily="Computer Modern")
savefig("k_ratios_Pt.png")

heatmap(range(-700u"nm", 700u"nm", 70), range(-700u"nm", 700u"nm", 70), res_Cu, color=cgrad([:black, :blue]), colorbar=false, aspect_ratio=:equal)
plot!(size=(300, 285), dpi=800, fontfamily="Computer Modern")
savefig("k_ratios_Cu.png")

surface(reshape(res, (70, 70)), color=:grays)
savefig("k-ratios.png")
# probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω->1.0, ϵ=ϵ->1.0)
# func = EPMAfem.interpolable(probe, pnsystem * discrete_rhs[1, 150, 1])

# dimless.((-1000.0u"nm", 0.0u"nm", -800u"nm", 800u"nm"), eq.dim_basis)
# contourf(-1.0:0.001:1.0, -1.4:0.001:0.0, (x, z) -> func(Point(z, x)), swapxy=true, aspect_ratio=:equal)

# x = [0.892174231487772, 2.9191310716712, 4.67217327615489, 5.86347735861073, 7.02000015922215, 8.1269555133322, 9.44869517450747, 10.4069561502208, 11.1504344110904, 12.2904347295346, 13.4469559379246, 14.041738228176, 15.0000007961107, 16.0573918881624, 17.0321721615999, 18.1391291079314, 18.9817390773607, 20.0060852050781, 21.0469554071841, 22.0713047193444, 22.8478247601053, 23.9382608164912, 25.028696872877, 26.1026088548743, 27.0608682383662, 28.1678251846977, 28.8452167013417, 29.9686945376189, 30.596520199983, 30.9930438497792, 31.9678273076596, 32.6452156398607, 33.4217388650645, 34.2973906143852, 35.0408688752548, 36.0486941130265, 37.0565225352412, 38.2295662257982, 38.9069577424423, 40.0799982485564, 42.1617418372113, 43.9626080056895, 45.0695649520211, 46.8043475607167, 50.0260866911515, 50.895652173913, 51.18260630732, 52.3060841435972, 54.1730402407439, 55.9078228494395, 57.6426054581352, 59.1295619798743, 60.8052214249321, 62.0652163298234, 63.1852167544158, 64.0052171790081]
# x = (x.*(2*1200.0/64.0) .- 1200.0)u"nm"
# y = [0.465221333053877, 0.468269231876545, 0.465221333053877, 0.465384617599244, 0.468486935376083, 0.465039910702643, 0.464810109795957, 0.465844219412617, 0.46664852258602, 0.464465413972498, 0.461707792019117, 0.461363085122517, 0.466188920772647, 0.464235601992669, 0.46768263220268, 0.463776000179296, 0.467912433109367, 0.468831642272684, 0.46664852258602, 0.46837203492274, 0.46458030888927, 0.462626995645864, 0.468831642272684, 0.471359457782806, 0.475725691619563, 0.486066765639882, 0.490892595753441, 0.501463476217018, 0.488249879789975, 0.482504840513099, 0.427467346191406, 0.364961294719547, 0.301191338260911, 0.249256178325769, 0.222599184564997, 0.213177314171424, 0.21030478622813, 0.212832607274824, 0.218003144284983, 0.218347851181584, 0.227539909595329, 0.23546807963201, 0.235697869465555, 0.241672715185689, 0.243970724252554, 0.243970724252554, 0.247417748925994, 0.245004844942356, 0.249026366345941, 0.250864773599434, 0.249945569972687, 0.251669082309407, 0.251560227791353, 0.253229326710473, 0.250169330480656, 0.252533866981983]

