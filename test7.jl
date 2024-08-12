using ConcreteStructs
using Zygote
using Lux
using Random
rng = Xoshiro(0)

model = Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1, tanh))

ps, st = Lux.setup(rng, model)

y, st = Lux.apply(model, [1.0, 1.0]', ps, st)

function observation(model, ps)
    xs = ran
end

@concrete struct PNProblem
    mass_concentrations
    specific_mass_attenuations
end

function _mass_concentrations(x, y, z, i)
    if abs(x) < 0.2
        return (1.0, 0.0, 0.0)[i]
    else
        return (0.0, 1.2, 0.3)[i]
    end
end

function _specific_mass_attenuations(ϵ, i)
    return exp(ϵ*(0.4, 0.5, 0.4)[i])
end

function number_of_densities(::PNProblem)
    return 3
end

prob = PNProblem(_mass_concentrations, _specific_mass_attenuations)

function mass_attenuations(prob)
    mass_attenuations = 0.0
    for i in 1:number_of_densities(prob)
        mass_attenuations += prob.mass_concentrations(0.0, 0.0, 0.0, i) * prob.specific_mass_attenuations(0.0, i)
    end
    return mass_attenuations
    # sum(prob.mass_concentrations(0.0, 0.0, 0.0, i) * prob.specific_mass_attenuations(0.0, i) for i in 1:number_of_densities(prob))
end

using BenchmarkTools
@benchmark mass_attenuations($prob)

using Serialization
dict = deserialize("boundary_matrix_dict.jls")
using MathLink

## this is not the wikipedia definition of real spherical harmonics, we add the (-1)^m to be consistent with the definition in https://arxiv.org/pdf/1410.1748
realsphericalharmonics = W`RealSphericalHarmonicY[l_, m_, \[Theta]_, \[Phi]_] := FullSimplify[If[m < 0, 
  (-1)^m*I/Sqrt[2]*(SphericalHarmonicY[l, m, \[Theta], \[Phi]] - (-1)^m*SphericalHarmonicY[l, -m, \[Theta], \[Phi]]),
  If[m > 0,
   (-1)^m*1/Sqrt[2]*(SphericalHarmonicY[l, -m, \[Theta], \[Phi]] + (-1)^m*SphericalHarmonicY[l, m, \[Theta], \[Phi]]),
   SphericalHarmonicY[l, m, \[Theta], \[Phi]]]]]`

x_boundary = W`N[Integrate[
    Abs[Sin[\[Theta]]*Cos[\[Phi]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
     RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
     Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]]`

y_boundary = W`N[Integrate[
 Abs[Sin[\[Theta]]*Sin[\[Phi]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
  RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
  Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]]`

z_boundary = W`N[Integrate[
 Abs[Cos[\[Theta]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
  RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
  Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]]`



weval(realsphericalharmonics)

using SphericalHarmonics

function eval_mathlink(m, θ, φ)
    return weval(W`N[RealSphericalHarmonicY[l, m, t, p]]`, l=m[1], m=m[2], t=θ, p=φ)
end

function eval_julia(m, θ, φ)
    return SphericalHarmonics.sphericalharmonic(θ,φ, m[1], m[2], SphericalHarmonics.RealHarmonics())
end

moms = SphericalHarmonicsMatrices.get_moments(2, Val(3))
eval_mathlink.(moms, 0.1, 0.1)
eval_julia.(moms, 0.1, 0.1)

evens = [m for m in SphericalHarmonicsMatrices.get_moments(5, Val(3)) if SphericalHarmonicsMatrices.is_even(m...)]

A = zeros(length(evens), length(evens))
for (i, m1) in enumerate(evens)
    for (j, m2) in enumerate(evens)
        val = weval(y_boundary; l1=m1[1], m1=m1[2], l2=m2[1], m2=m2[2])
        @show val
        A[i, j] = val
    end
end
A
round.(assemble_boundary_matrix(5, Val(2), :pp, Val(3)), digits=8)

compute_boundary_matrix_entry_new(Val(2), evens[2], evens[1])
SphericalHarmonicsMatrices.compute_boundary_matrix_entry(2, evens[2], evens[1], 1e-11)

A

weval(W`Sin[x]`; x=1.0)

A = problem.ρp[1]

B = zeros(10011, 350)
C = rand(10011, 350)

@benchmkar mul!(B, A, C)