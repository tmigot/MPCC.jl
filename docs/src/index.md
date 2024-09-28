```@meta
CurrentModule = MPCC
```

# MPCC

Documentation for [MPCC](https://github.com/tmigot/MPCC.jl).

# MPCC: Optimization Models with Complementarity Constraints

Set of tools to model nonlinear continuous optimization model

```math
    \min f(x) \text{ s.t. } l ≤ c(x) ≤ u,
```

extended with complementarity constraints:

```math
    G(x) ≥ 0, H(x) ≥ 0, G(x) ∘ H(x) ≤ 0.
```

The resulting model is called an MPCC.

This package also handles vanishing constraints (MPVC)

```math
    H(x) ≥ 0, G(x) ∘ H(x) ≤ 0,
```

kink constraints

```math
    H(x) ≥ 0, G(x) ∘ H(x) = 0,
```

and switching constraints

```math
    G(x) ∘ H(x) = 0.
```

Denote ∘ the componentwise product of two vectors.

This package extends the NLPModel API defined in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
The role of NLPModels.jl is to define an API that users and solvers can rely on.

The package contains the basic tools to use the [Stopping](https://github.com/SolverStoppingJulia/Stopping.jl) framework, and exports an `MPCCStopping`.

## Installation

```julia
] add MPCC
```

## Example

The simplest way to define an `MPCCModel` uses automatic differentiation:

```julia
using MPCC
f(x) = sum(x) # objective function
x0 = ones(6)
G(x) = [x[1]; x[3]]
H(x) = [x[2]; x[4]]
lccg, lcch = zeros(2), zeros(2)
lvar, uvar = fill(-10.0, size(x0)), fill(10.0, size(x0))
admpcc = ADMPCCModel(G, H, lccg, lcch, f, x0, lvar, uvar)
```

The resulting is an instance of an `AbstractMPCCModel` with all capabilities of an `AbstractNLPModel` such as

```julia
x = rand(6)
grad(admpcc, x) # returns the gradient of the objective function
neval_grad(admpcc) # returns the number of evaluation of grad
reset!(admpcc) # reset the internal counter
```

but also exports functions to deal with complementarity constraints

```julia
x = rand(6)
(consG(admpcc, x), consH(admpcc, x))
```

but also `jac_G_structure`, `jac_H_structure`, `jac_G_coord`, `jac_H_coord`, `jGprod`, `jHprod`, `jGtprod`, `jHtprod`, `hGprod`, `hHprod`.

It also possible to convert the problem as a classical nonlinear optimization model treating the complementarity constraint as a nonlinear inequalities

```math
    H(x) ≥ 0, G(x) ≥ 0, G(x) ∘ H(x) ≤ 0,
```

using `NLMPCC` as follows

```julia
nlp = NLMPCC(admpcc)
```

so that

```julia
cons(nlp, x)
```

returns a vector consisting of `[c(x), G(x), H(x), G(x) ∘ H(x)]`.

## Contributors

```@raw html
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://tmigot.github.io"><img src="https://avatars.githubusercontent.com/u/25304288?v=4?s=100" width="100px;" alt="Tangi Migot"/><br /><sub><b>Tangi Migot</b></sub></a><br /><a href="#projectManagement-tmigot" title="Project Management">📆</a> <a href="#maintenance-tmigot" title="Maintenance">🚧</a> <a href="#doc-tmigot" title="Documentation">📖</a> <a href="#test-tmigot" title="Tests">⚠️</a> <a href="#code-tmigot" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
```
