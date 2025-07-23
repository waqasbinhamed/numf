# NuMF

This repository provides a Python implementation of Nonnegative Unimodal Matrix Factorization (NuMF), for matrix factorization algorithm that enforces both nonnegativity and unimodality constraints on the factors. NuMF is particularly useful for applications where the underlying data is expected to have a single peak (unimodal structure) and nonnegative values, such as in signal processing, bioinformatics, and chemometrics.

The implementation is based on the research paper ["Nonnegative Unimodal Matrix Factorization"](https://ieeexplore.ieee.org/iel7/9413349/9413350/09414631.pdf).

## Installation

1. Clone the repository.
2. In the main project directory named `numf`, run:
   ```bash
   pip install -r requirements.txt
   ```
