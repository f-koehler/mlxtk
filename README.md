# ML-X Toolkit (mlxtk)

The ML-X Toolkit (mlxtk) provides utilities to easily setup, execute and analyse simulations using the Hamburg ML-MCTDH(X) (multi-layer multi-configuration time-dependent hartree) implementation to study the dynamics as well as the static properties of quantum systems.
Distinguishable, fermionic and bosonic degrees as well as mixtures thereof are within the scope of this method.

## Features
This list is incomplete at the moment.

- simulation tasks:
  - [x] create complex simulations from building blocks (tasks)
  - [x] use hashes to determine which task have already been executed
  - [x] store timings for each task
  - [x] store parameters
  - [x] detect parameter changes
  - implemented tasks:
    - [x] wave function creation
    - [x] operator creation
    - [x] propagation
    - [x] relaxation
    - [ ] improved relaxation
    - [x] expectation values
    - [x] operator variances
    - [x] fixed number state projection
- parameter scans:
  - [x] cartesian product of parameter sets
  - [x] filters on the parameter combinations
  - [x] detection of added/removed combinations (including the renaming of remaining simulations)
  - [x] dry-run to detect which simulations would be run
  - [x] execution of a specific simulation
  - [x] submission to SGE batch system
  - [x] (parallel) local execution
  - [x] tabular output for parameter combinations
  - [x] generation of AFS sync scripts for SGE batch system
  - [x] scan view GUI to easily create plots for each combination
- DVR interface:
  - [x] access to QDTK DVRs
  - [x] memoization of already created DVRs
- HDF5 conversion of all major output files
- plotting:
  - [x] classes to create Qt5 based plotting GUIs easily
  - [x] easy integration with Jupyter notebooks (needs some refactoring)
  - plotting of observables:
    - [x] energy (difference)
    - [x] expectation values (difference)
    - [x] one body density (difference) as heatmap
    - [x] one body density for each time step (slider)
    - [x] norm (difference)
    - [x] SPF overlap (difference)
    - [x] natural populations
