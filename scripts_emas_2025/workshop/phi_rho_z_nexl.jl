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

NeXLCore.edgeenergy(atomic_subshell::NeXLCore.AtomicSubShell) = NeXLCore.edgeenergy(z(atomic_subshell), shell(atomic_subshell).n)

function compute_phi_z_epmafem(atomic_subshell, beam_energy, depth=1000u"nm")
	mat = [element(atomic_subshell)]
	ϵ_range = range(edgeenergy(atomic_subshell)u"eV" - 1u"keV", beam_energy+5.0u"keV", length=200)
    eq = NExt.epma_equations(mat, ϵ_range, 27)
	
    model = NExt.epma_model(eq, (-depth, 0.0u"nm"), (100), 27)
	arch = EPMAfem.cpu()
	pnproblem = EPMAfem.discretize_problem(eq, model, arch, updatable=false)

	discrete_system = EPMAfem.implicit_midpoint(pnproblem, EPMAfem.PNSchurSolver)

	excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [dimless(beam_energy, eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_energy_σ=0.05, beam_direction_κ=200.0)
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

	func = EPMAfem.interpolable(probe, discrete_system * discrete_rhs[1])
	fac =  phi0 / func(Point(0.0))
	return z -> func(Point(dimless(z, eq.dim_basis))) * fac
end
