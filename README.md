# DMET

Density Matrix Embedding Theory (DMET) implementation for investigating strongly correlated molecular systems.

This repository is based on the original [QC-DMET](https://github.com/SebWouters/QC-DMET) implementation developed by Sebastian Wouters, extended to interface with the block2 DMRG solver and to facilitate calculations on high-performance computing (HPC) systems.

## Features

- Density Matrix Embedding Theory for molecular systems.
- Integration with PySCF.
- Integration with the block2 DMRG solver.
- Containerized execution using Docker and Apptainer.

## Dependencies

- [PySCF](https://github.com/pyscf/pyscf)
- [block2](https://github.com/block-hczhai/block2-preview)
- NumPy
- SciPy

## Installation

The repository includes a Dockerfile for building a containerized environment containing the required software dependencies. The resulting Docker image can also be converted into an Apptainer-compatible SIF image for execution on HPC systems.

## References

The original DMET implementation is described in:

S. Wouters, C. A. Jiménez-Hoyos, Q. Sun, and G. K.-L. Chan, *A Practical Guide to Density Matrix Embedding Theory in Quantum Chemistry*, J. Chem. Theory Comput. **12**, 2706–2719 (2016).

DOI: https://doi.org/10.1021/acs.jctc.6b00316

## Copyright and license

Original implementation:

Copyright (C) 2015 Sebastian Wouters.

Modifications and extensions:

Copyright (C) 2026 Quentin Staelens.

This software is distributed under the GNU General Public License, version 2 or any subsequent version (GPL-2.0-or-later).

The software is provided without warranty. See the LICENSE file for the complete license terms.
