# Nonnegative Unimodal Matrix Factorization

This repository provides a Python implementation of Nonnegative Unimodal Matrix Factorization (NuMF), for matrix factorization algorithm that enforces both nonnegativity and unimodality constraints on the factors. NuMF is particularly useful for applications where the underlying data is expected to have a single peak (unimodal structure) and nonnegative values, such as in signal processing, bioinformatics, and chemometrics.

The implementation is based on the research paper ["Nonnegative Unimodal Matrix Factorization"](https://ieeexplore.ieee.org/iel7/9413349/9413350/09414631.pdf).

## Notebooks
- **`notebooks/EDA and data preparation.ipynb`**: Performs exploratory data analysis (EDA) and creates the cleaned dataset used in the `notebooks/NuMF on Canada COVID data.ipynb` notebook.
- **`notebooks/NuMF on Canada COVID data.ipynb`**: Demonstrates the use of NuMF on Canada's COVID-19 data.
- **`notebooks/Toy Example.ipynb`**: Provides a simple synthetic example to illustrate the basic functionality and behavior of NuMF. This notebook also demonstrates the effects of multigrid acceleration and regularization on the factorization results.

## Data Sources (in `data/`)

The datasets used in this repository are obtained from reputable public sources:

- **COVID-19 Variant Data:**  
  Downloaded from [GISAID](https://gisaid.org), using their Excel variant statistics files. 

- **Provincial Case Data:**  
  Sourced from the [Johns Hopkins University CSSE COVID-19 Data Repository](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv).  
  This includes cumulative and daily case counts for Canadian provinces.

- **National Cases, Hospitalizations, and Deaths:**  
  Downloaded from [Our World In Data (OWID)](https://ourworldindata.org/covid-cases).  
  This includes national-level time series for cases, hospitalizations, and deaths.
