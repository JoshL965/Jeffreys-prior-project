# Jeffreys Prior Project

A Python-based cosmological inference project for computing likelihoods, Fisher matrices, and Jeffreys priors across multiple observational datasets.

This repository combines cosmological datasets such as DESI BAO, DES supernovae, Pantheon+, Union3, and Planck CMB data with JAX-based emulator predictions to perform parameter inference and statistical analysis in a flexible, extensible framework.

## Why this project exists

The project is designed to study how cosmological parameters are constrained by data and how prior knowledge affects inference. In particular, it implements a Jeffreys prior framework based on the Fisher information matrix, which is useful when exploring parameter-space geometry and noninformative priors in cosmological model comparison.

The code is intended for:

- combining multiple cosmological probes into a single analysis
- evaluating likelihoods and chi-squared statistics for model fits
- computing Fisher matrices and Jeffreys priors
- loading pre-trained cosmological emulators for fast theory predictions
- producing posterior-style plots and diagnostics

## Repository layout

```text
Jeffreys-prior-project/
├── Data_for_class/
│   ├── DESI_BAO/
│   │   ├── DR1/
│   │   └── DR2/
│   ├── DES_Y5/
│   ├── DES_Doveckie/
│   ├── PantheonPlus/
│   ├── Planck_2018_low_ell/
│   ├── Planck_2018_plik_lite/
│   ├── Union3/
│   └── ...
├── Emulators/
│   ├── Base/
│   ├── Curvature/
│   └── Neutrino_mass/
├── pipeline/
│   └── jp_class.py
├── PROJECT_OVERVIEW.md
├── SETUP.md
├── README.md
└── .gitignore
```

## Main components

### pipeline/jp_class.py

This is the core analysis module. It contains:

- dataset loaders for BAO, SN, and CMB products
- theory prediction functions for each dataset
- emulator-loading logic for JAX-based cosmological emulators
- the `stats_quantities` class that bundles the statistical tools
- likelihood, chi-square, Fisher matrix, and Jeffreys prior calculations
- plotting routines for GetDist-based triangle plots

### Data_for_class/

This directory contains the observational data used by the project. Data are organized by survey or experiment and are expected to follow the file naming and structure assumed by the loader functions in `jp_class.py`.

### Emulators/

This directory contains emulator files organized by parameter-extension family, such as:

- Base
- Curvature
- Neutrino_mass

The emulator files are used to approximate theory predictions much faster than full cosmological solvers.

## Supported datasets

The project is set up to work with the following observational datasets:

- BAO_DR1
- BAO_DR2
- DES_Y5
- DES_Doveckie
- Union3
- CMB_plik_lite
- CMB_high_ell_TTTEEE
- CMB_low_ell_TT
- CMB_low_ell_EE
- PantheonPlus

## Supported parameter spaces

The code includes several model families:

- Base
  - `ombh2`, `omch2`, `h`, `ns`, `logA`, `tau`, `w0`, `wa`
- Curvature
  - includes `omk`
- Neutrino_mass
  - includes `mnu`

## Scientific focus

This repository is focused on cosmological parameter inference using Jeffreys priors. In practice, the workflow is:

1. load a set of datasets
2. evaluate their corresponding theory vector using emulator-backed predictions
3. compute the residual between data and theory
4. build the inverse covariance-weighted likelihood
5. estimate Fisher information from parameter derivatives
6. compute a Jeffreys prior from the Fisher matrix
7. use these ingredients for parameter estimation and model comparison

This is especially useful in cosmology where different parameter combinations can be strongly degenerate and where the geometry of the parameter space matters for prior choice.

## Requirements

### Python

Python 3.8+ is recommended.

### Dependencies

Install the required packages:

```bash
pip install numpy jax pandas scipy matplotlib getdist cosmopower-jax
```

If you want GPU-enabled JAX support, install the appropriate JAX build for your system, for example:

```bash
pip install "jax[cuda11_cudnn82]"
```

or use the CPU-only version if that matches your setup.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/JoshL965/Jeffreys-prior-project.git
cd Jeffreys-prior-project
```

2. Ensure the repository is on your Python path if needed:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

3. Confirm the data and emulator folders exist in the expected locations:

```bash
ls Data_for_class
ls Emulators
```

## Quick start

You can import the analysis class directly:

```python
from pipeline.jp_class import stats_quantities

requested_data = ['DES_Y5']
extensions = 'Base'

stats = stats_quantities(requested_data, extensions)
```

To evaluate a likelihood at a parameter point:

```python
params = {
    'ombh2': 0.022,
    'omch2': 0.12,
    'h': 0.67,
    'ns': 0.96,
    'logA': 3.05,
    'tau': 0.06,
    'w0': -1.0,
    'wa': 0.0,
    'M_b': -19.3,
    'A_planck': 1.0,
}

log_like = stats.log_likelihood(params)
chi2 = stats.chi_square(params)
print(log_like)
print(chi2)
```

## File conventions

The project assumes the following local organization:

- data files live under `Data_for_class/<dataset>`
- emulator files live under `Emulators/<parameter_space>/`
- the code resolves those paths relative to the repository root using Python's `os.path` machinery

This avoids depending on machine-specific absolute paths such as `/cephfs/...`.

## Usage notes

The `stats_quantities` class exposes the following main methods:

- `log_likelihood(params)`
- `chi_square(params)`
- `theory(params)`
- `Fisher_matrix(params, sampled_params)`
- `Jeffreys_prior(params, sampled_params)`
- `MAP(limits, init_guess, param_names)`
- `getdist_plot(...)`

These methods allow you to evaluate the model, compute constraints, and create analysis plots.

## Troubleshooting

### Import errors

If you see a `ModuleNotFoundError`, run the project from the repository root or add it to `PYTHONPATH`.

### Missing data files

Check that the directory structure below `Data_for_class` matches the dataset names expected by the loader functions in `pipeline/jp_class.py`.

### Missing emulator files

Ensure the emulator `.npz` files are present under the correct parameter-space subdirectory inside `Emulators/`.

### JAX issues

If JAX fails to initialize or detect hardware, confirm your JAX installation matches your system and CUDA setup.

## Additional documentation

For more context on the scientific motivation and project design, see:

- `PROJECT_OVERVIEW.md`
- `SETUP.md`

## License

This repository does not currently declare a license in the root metadata. If you plan to distribute or reuse it publicly, consider adding an explicit license file such as MIT or Apache 2.0.

## Contact

This repository is maintained by the project owner under the GitHub user `JoshL965`.
