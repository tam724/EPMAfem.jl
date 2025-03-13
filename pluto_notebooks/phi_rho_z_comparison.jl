### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3dca95fc-f545-11ef-3897-3312f7e6c36b
begin
	using Pkg
	Pkg.add(url="https://github.com/tam724/EPMAfem.jl", rev="6f172a6")
	Pkg.add("NeXLCore")
	Pkg.add("NeXLMatrixCorrection")
	Pkg.add("Unitful")
	Pkg.add("Dimensionless")
	Pkg.add("Plots")
end

# ╔═╡ e4e75e53-4194-4491-ae9f-6c0422c123cd
begin
	using EPMAfem, NeXLCore, NeXLMatrixCorrection, Unitful, Dimensionless, Plots
	using EPMAfem.Gridap
	NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)
end

# ╔═╡ ed4b009c-7a77-45ba-86e0-c1a51c266360
html"""<style>
main {
    max-width: 1000px;
}
"""

# ╔═╡ 50cb20f8-e609-4bbf-a867-a35214ed652b
NeXLCore.edgeenergy(atomic_subshell::NeXLCore.AtomicSubShell) = NeXLCore.edgeenergy(z(atomic_subshell), shell(atomic_subshell).n)

# ╔═╡ f534f333-f7ee-44ba-818e-a883cdc122b1
function compute_phi_z_pn(atomic_subshell, beam_energy)
	mat = [element(atomic_subshell)]
	ϵ_range = range(edgeenergy(atomic_subshell)u"eV" - 1u"keV", beam_energy+5.0u"keV", length=200)
	eq = NExt.epma_equations(mat, ϵ_range, 23)
	
	space_extensions_dimless = dimless.((-1000u"nm", 0u"nm"), eq.dim_basis)
	space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel(space_extensions_dimless, (200)))
	energy_model = dimless(ϵ_range, eq.dim_basis)
	direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(23, 1)

	model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
	arch = EPMAfem.cpu()
	pnproblem = EPMAfem.discretize_problem(eq, model, arch, updatable=false)

	discrete_system = EPMAfem.schurimplicitmidpointsystem(pnproblem)

	excitation = EPMAfem.PNExcitation([(x=0.0, y=0.0)], [dimless(beam_energy, eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)])
	# influx = EPMAfem.compute_influx(excitation, model)
	influx_weighted = EPMAfem.compute_influx(excitation, model, ϵ -> 1e15*dimless(ionizationcrosssection(atomic_subshell, dimful(ϵ, u"eV", eq.dim_basis) / u"eV" |> upreferred)u"cm"^2, eq.dim_basis))

	discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
	# discrete_outflux = EPMAfem.discretize_outflux(model, EPMAfem.cuda())
	discrete_outflux_weighted = EPMAfem.discretize_outflux(model, arch, ϵ -> 1e15*dimless(ionizationcrosssection(atomic_subshell, dimful(ϵ, u"eV", eq.dim_basis) / u"eV" |> upreferred)u"cm"^2, eq.dim_basis))

	# outflux = 2*discrete_outflux * discrete_system * discrete_rhs[1] + influx[1]
	outflux_weighted = 2*discrete_outflux_weighted * discrete_system * discrete_rhs[1] + influx_weighted[1]

	# η_BS = outflux / -influx[1]

	phi0 = 1.0 + outflux_weighted / -influx_weighted[1]


	probe = EPMAfem.PNProbe(model, arch; Ω=Ω->1.0, ϵ=ϵ -> dimless(ionizationcrosssection(atomic_subshell, dimful(ϵ, u"eV", eq.dim_basis) / u"eV" |> upreferred)u"cm"^2, eq.dim_basis))

	res = probe(discrete_system * discrete_rhs[1])

	func = EPMAfem.SpaceModels.interpolable(res, space_model)
	fac =  phi0 / func(Point(0.0))
	return z -> func(Point(dimless(z, eq.dim_basis))) * fac
end

# ╔═╡ f5848706-46d0-48da-a683-740a44c5210e
function ρz(elm, z)
	val = ustrip(u"g"/u"cm"^2, elm.density * z)
	return val
end

# ╔═╡ 368fc8cc-08ea-4eec-9c7e-5e8859d31a2c
function compute_phi_z_nexl(atomic_subshell, beam_energy, alg=XPP)
	nexl_alg = alg(Material("", Dict(element(atomic_subshell)=>1.0)), atomic_subshell, ustrip(u"eV", beam_energy))
	# ρz(z) = ustrip(u"g"/u"cm"^2, element(atomic_subshell).density * z)
	return z -> ϕ(nexl_alg, -ρz(element(atomic_subshell), z))
end

# ╔═╡ 3327b627-a690-4815-bbdc-a666f483c55a
begin
	atomic_subshell = n"Cr K"
	beam_energy = 10.0u"keV"
	ϕz_pn = compute_phi_z_pn(atomic_subshell, beam_energy)
	ϕz_nexl_XPP = compute_phi_z_nexl(atomic_subshell, beam_energy, XPP)
	ϕz_nexl_cit = compute_phi_z_nexl(atomic_subshell, beam_energy, CitZAF)
	ϕz_nexl_xphi = compute_phi_z_nexl(atomic_subshell, beam_energy, XPhi)
	z_range = -700.0u"nm":1u"nm":0.0u"nm"
	
	plot(z_range, ϕz_pn, label="PN")
	plot!(z_range,ϕz_nexl_XPP, label="PAP")
	plot!(z_range,ϕz_nexl_cit, label="CitZAF")
	plot!(z_range,ϕz_nexl_xphi, label="XPhi")
end

# ╔═╡ Cell order:
# ╠═ed4b009c-7a77-45ba-86e0-c1a51c266360
# ╠═3dca95fc-f545-11ef-3897-3312f7e6c36b
# ╠═e4e75e53-4194-4491-ae9f-6c0422c123cd
# ╠═50cb20f8-e609-4bbf-a867-a35214ed652b
# ╠═f534f333-f7ee-44ba-818e-a883cdc122b1
# ╠═f5848706-46d0-48da-a683-740a44c5210e
# ╠═368fc8cc-08ea-4eec-9c7e-5e8859d31a2c
# ╠═3327b627-a690-4815-bbdc-a666f483c55a
