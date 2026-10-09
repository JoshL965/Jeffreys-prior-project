# Jeffreys Prior Project Overview

## Purpose

This project implements a cosmological parameter inference framework using **Jeffreys priors** in combination with multiple astrophysical datasets. The primary goal is to perform Bayesian inference on cosmological models while quantifying prior information through the Fisher information matrix.

## Scientific Context

### What are Jeffreys Priors?

Jeffreys priors are non-informative Bayesian priors derived from the Fisher information matrix of a model. They are defined as:

$$p(\boldsymbol{\theta}) \propto \sqrt{\det(F(\boldsymbol{\theta}))}$$

where $F(\boldsymbol{\theta})$ is the Fisher information matrix. This prior:
- Automatically adapts to the parameter space geometry
- Provides scale-invariant inference (invariant under reparameterization)
- Encodes information about which parameters are well-constrained by the data

### Cosmological Application

In cosmology, understanding parameter constraints from diverse observational datasets is essential for model validation. This project leverages:

- **Cosmic Microwave Background (CMB)** data from Planck 2018 to constrain early-universe physics
- **Baryon Acoustic Oscillations (BAO)** from DESI to probe large-scale structure and expansion history
- **Type Ia Supernovae (SNe)** from multiple surveys to measure expansion acceleration

These datasets provide complementary constraints on cosmological parameters describing:
- Matter content and composition
- Expansion history and dark energy
- Neutrino properties and spatial curvature

## Key Features

### 1. Multi-Dataset Analysis
- Seamlessly combine constraints from CMB, BAO, and SNe observations
- Load and process Planck 2018, DESI DR1/DR2, DES, Union3, and Pantheon+ datasets
- Flexible dataset selection for targeted studies

### 2. Fisher Information Framework
- Compute the Fisher information matrix from observational likelihoods
- Extract parameter degeneracies and constraints
- Derive Jeffreys priors automatically

### 3. Statistical Tools
- **Log-likelihood calculation**: Evaluate likelihood for any cosmological model
- **χ² analysis**: Standard chi-squared goodness-of-fit metrics
- **MAP estimation**: Maximum A Posteriori point finding via L-BFGS-B optimization
- **Plotting utilities**: Integration with GetDist for triangle plots and posterior visualization

### 4. Cosmological Emulators
- Uses pre-trained neural network emulators (CosmoPower-JAX) for rapid theory predictions
- Emulators for CMB power spectra (TT, TE, EE), background expansion (H, D_A), and derived quantities
- Accelerates parameter space exploration and MCMC sampling

## Architecture

### Core Components

**`stats_quantities` Class**
The main interface for cosmological analysis. Provides methods for:
- Loading and managing multiple datasets
- Computing likelihoods and chi-squared statistics
- Calculating Fisher matrices and Jeffreys priors
- Optimizing to the maximum a posteriori (MAP) point
- Generating publication-quality plots

**Dataset Catalog**
Pre-configured loading and theory prediction pipelines for supported datasets:
- Planck CMB (multiple multipole ranges)
- DESI BAO (DR1 and DR2)
- DES/Union3/Pantheon+ Supernovae

**Theory Prediction Functions**
Dataset-specific functions that:
- Accept parameter dictionaries
- Use cosmological emulators for rapid computations
- Return predictions in the data space
- Handle unit conversions and binning

### Data Flow

```
Parameter Dictionary
        ↓
    Emulators (JAX-based)
        ↓
    Theory Predictions
        ↓
    Comparison to Data
        ↓
    Likelihood / χ²
        ↓
    Fisher Matrix / Jeffreys Prior
```

## Supported Models

### Parameter Spaces

1. **Base ΛCDM + Dark Energy**
   - 8 parameters: baryon density, cold dark matter density, Hubble constant, scalar index, amplitude, optical depth, CPL dark energy parameters
   - Standard cosmological model

2. **Open ΛCDM**
   - Adds spatial curvature as additional parameter
   - Explores departures from flat universe assumption

3. **Extended Neutrino Physics**
   - Adds total neutrino mass as parameter
   - Probes neutrino mass hierarchy constraints

## Use Cases

### 1. Constraint Comparison
Compare how different datasets constrain cosmological parameters:
```python
stats_bao = stats_quantities(['BAO_DR2'], 'Base')
stats_cmb = stats_quantities(['CMB_high_ell_TTTEEE'], 'Base')
stats_combined = stats_quantities(['BAO_DR2', 'CMB_high_ell_TTTEEE'], 'Base')
```

### 2. Prior Studies
Evaluate how Jeffreys priors affect parameter estimation:
```python
fisher = stats.Fisher_matrix(params, param_names)
jeffreys_log_prior = stats.Jeffreys_prior(params, param_names)
```

### 3. Sensitivity Analysis
Understand degeneracies through Fisher matrix eigenvalues/eigenvectors:
```python
eigenvalues = jnp.linalg.eigvalsh(fisher)
```

### 4. Model Selection
Use likelihoods for Bayesian model comparison across different parameter spaces.

## Technical Implementation Details

### JAX Integration
- All numerical operations use JAX for automatic differentiation (AD)
- Gradients computed via `jax.jacfwd()` for Fisher matrix calculations
- Just-In-Time (JIT) compilation support for performance optimization
- 64-bit precision enabled for numerical stability

### Cosmological Emulators
- **CosmoPower-JAX**: Fast neural network emulators pre-trained on CLASS outputs
- Multiple probe types: CMB power spectra, background quantities
- Redshift grid: 0 to 5.0 with 1000 points for interpolation

### Optimization
- L-BFGS-B algorithm for MAP estimation
- Automatic parameter scaling for numerical stability
- Includes A_planck calibration prior (Gaussian, σ=0.0025)

### Visualization
- Integration with GetDist for MCMC sample plotting
- Triangle plots with customizable parameter limits
- Adds reference lines for ΛCDM and MAP points
- Publication-ready formatting options

## Mathematical Foundations

### Likelihood Function
$$\ln L(\boldsymbol{\theta}) = -\frac{1}{2}(\mathbf{d} - \mathbf{t}(\boldsymbol{\theta}))^T \Sigma^{-1} (\mathbf{d} - \mathbf{t}(\boldsymbol{\theta}))$$

where:
- $\mathbf{d}$: observational data vector
- $\mathbf{t}(\boldsymbol{\theta})$: theory predictions
- $\Sigma$: covariance matrix

### Fisher Information Matrix
$$F_{ij} = -\left\langle \frac{\partial^2 \ln L}{\partial \theta_i \partial \theta_j} \right\rangle$$

Estimated from Jacobian of theory predictions:
$$F = \mathbf{J}^T \Sigma^{-1} \mathbf{J}$$

### Jeffreys Prior
$$\ln p_J(\boldsymbol{\theta}) = \frac{1}{2} \ln \det F(\boldsymbol{\theta})$$

## Dependencies

- **JAX** (0.3.0+): Automatic differentiation and numerical computing
- **NumPy**: Classical numerical operations
- **Pandas**: Data file I/O
- **SciPy**: Optimization and special functions
- **CosmoPower-JAX**: Cosmological emulator interface
- **GetDist**: MCMC sample analysis and visualization
- **Matplotlib**: Plotting backend

## References

**Jeffreys Priors in Cosmology:**
- Jeffreys, H. (1961). Theory of Probability. Oxford University Press.
- Heavens, A. F., Kitching, T. D., & Verde, L. (2011). On model selection forecasting, dark energy, and modified gravity. Journal of Cosmology and Astroparticle Physics.

**Datasets:**
- Planck Collaboration et al. (2018): Planck 2018 results
- DESI Collaboration et al. (2024): DESI BAO measurements
- DES Collaboration et al.: Dark Energy Survey SN sample

**Emulators:**
- Spurio Mancini, A., Piras, D., et al. (2022). CosmoPower: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys. Monthly Notices of the Royal Astronomical Society.

## License

[Specify your license here]

## Contact

For questions or issues, please contact the repository maintainer.
