# MPCC

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tmigot.github.io/MPCC.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://tmigot.github.io/MPCC.jl/dev)
[![Build Status](https://github.com/tmigot/MPCC.jl/workflows/Test/badge.svg)](https://github.com/tmigot/MPCC.jl/actions)
[![Test workflow status](https://github.com/tmigot/MPCC.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/tmigot/MPCC.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/tmigot/MPCC.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/tmigot/MPCC.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/tmigot/MPCC.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/tmigot/MPCC.jl/actions/workflows/Docs.yml?query=branch%3Amain)

[![Coverage](https://codecov.io/gh/tmigot/MPCC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tmigot/MPCC.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/tmigot/MPCC.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)

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

## Example

The simplest way to define an `MPCCModel` uses automatic differentiation:

```
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

```
x = rand(6)
grad(admpcc, x) # returns the gradient of the objective function
neval_grad(admpcc) # returns the number of evaluation of grad
reset!(admpcc) # reset the internal counter
```

but also exports functions to deal with complementarity constraints

```
x = rand(6)
(consG(admpcc, x), consH(admpcc, x))
```

but also `jac_G_structure`, `jac_H_structure`, `jac_G_coord`, `jac_H_coord`, `jGprod`, `jHprod`, `jGtprod`, `jHtprod`, `hGprod`, `hHprod`.

It also possible to convert the problem as a classical nonlinear optimization model treating the complementarity constraint as a nonlinear inequalities

```math
    H(x) ≥ 0, G(x) ≥ 0, G(x) ∘ H(x) ≤ 0,
```

using `NLMPCC` as follows

```
nlp = NLMPCC(admpcc)
```

so that

```
cons(nlp, x)
```

returns a vector consisting of `[c(x), G(x), H(x), G(x) ∘ H(x)]`.

## How to Cite

If you use MPCC.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/tmigot/MPCC.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://tmigot.github.io/MPCC.jl/dev/90-contributing/).

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
