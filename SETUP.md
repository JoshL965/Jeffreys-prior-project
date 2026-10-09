# Setup Guide

## Overview

This project requires Python 3.8+ with several scientific computing dependencies. The setup process involves installing dependencies and configuring to use local data and emulator files included in the repository.

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- ~1-2 GB of disk space for data and emulator files (already included in the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JoshL965/Jeffreys-prior-project.git
cd Jeffreys-prior-project
```

### 2. Install Dependencies

The project requires the following Python packages:

```bash
pip install numpy jax jax[cuda11_cudnn82] jax-numpy pandas cosmopower-jax scipy getdist matplotlib
```

**Core Dependencies:**
- **NumPy**: Numerical computing library
- **JAX**: High-performance numerical computing with automatic differentiation and GPU support
- **JAX-NumPy**: JAX implementation of NumPy API
- **Pandas**: Data manipulation and CSV file handling
- **CosmoPower-JAX**: Cosmological emulator interface for JAX
- **SciPy**: Scientific computing functions (optimization, I/O)
- **GetDist**: MCMC sample plotting and analysis
- **Matplotlib**: Data visualization

### 3. Verify Installation

Create a simple test script to verify JAX is working:

```python
import jax
import jax.numpy as jnp
print(jax.devices())  # Should display available devices
```

## Project Structure

```
Jeffreys-prior-project/
├── pipeline/
│   └── jp_class.py                    # Main cosmological analysis module
├── Data_for_class/                    # Local data directory
│   ├── DESI_BAO/                      # DESI Baryon Acoustic Oscillation data
│   │   ├── DR1/
│   │   └── DR2/
│   ├── DES_Y5/                        # Dark Energy Survey supernovae
│   │   ├── DES-SN5YR_HD.csv
│   │   └── covsys_000.txt
│   ├── DES_Doveckie/                  # DES reanalyzed SN data
│   ├── Union3/                        # Union3 SN compilation
│   ├── PantheonPlus/                  # Pantheon+ SN sample
│   └── Planck_2018_*/                 # Planck CMB data
├── Emulators/                         # Cosmological neural network emulators
│   ├── Base/
│   ├── Curvature/
│   └── Neutrino_mass/
├── README.md
├── PROJECT_OVERVIEW.md
└── SETUP.md
```

## Using Local Data and Emulators

The code is now configured to use data and emulators from local directories within the repository. No external HPC filesystem access is required.

### File Paths Configuration

All file paths in `pipeline/jp_class.py` are now relative to the repository root:

- **Data files**: Located in `Data_for_class/<dataset_name>/`
- **Emulator files**: Located in `Emulators/<parameter_space>/`

### Current Supported Datasets

The following datasets are available in `Data_for_class/`:

| Dataset | Directory | Files |
|---------|-----------|-------|
| `DESI_BAO_DR1` | `Data_for_class/DESI_BAO/DR1` | data.txt, cov_mat.txt, model_order.txt |
| `DESI_BAO_DR2` | `Data_for_class/DESI_BAO/DR2` | data.txt, cov_mat.txt, model_order.txt |
| `DES_Y5` | `Data_for_class/DES_Y5` | DES-SN5YR_HD.csv, covsys_000.txt |
| `DES_Doveckie` | `Data_for_class/DES_Doveckie` | DES-Dovekie_HD.csv, STAT+SYS.npz |
| `Union3` | `Data_for_class/Union3` | lcparam_full.txt, mag_covmat.txt |
| `CMB_plik_lite` | `Data_for_class/Planck_2018_plik_lite` | c_matrix_plik_v22.dat, cl_cmb_plik_v22.dat, others |
| `CMB_low_ell_TT` | `Data_for_class/Planck_2018_low_ell` | CTT_bin_low_ell_2018.dat, blmin_low_ell.dat, others |
| `CMB_low_ell_EE` | `Data_for_class/Planck_2018_low_ell` | lognormal_fit_3bins_EE.txt, others |
| `PantheonPlus` | `Data_for_class/PantheonPlus` | Pantheon+SH0ES.dat, Pantheon+SH0ES_STAT+SYS.cov |

### Current Supported Parameter Spaces and Emulators

Emulators are organized by parameter space in the `Emulators/` directory:

1. **Base**: Standard 8-parameter flat ΛCDM + dark energy
   - Location: `Emulators/Base/`
   - Emulator files: `cmb_tt.npz`, `cmb_ee.npz`, `cmb_te.npz`, `background_H.npz`, `background_Da.npz`, `cmb_derived.npz`

2. **Curvature**: Base parameters plus spatial curvature
   - Location: `Emulators/Curvature/`
   - Same emulator files as Base

3. **Neutrino_mass**: Base parameters plus total neutrino mass
   - Location: `Emulators/Neutrino_mass/`
   - Same emulator files as Base

## Configuration

### JAX Configuration

The project enables 64-bit precision for numerical accuracy:

```python
jax.config.update("jax_enable_x64", True)
```

This setting is already configured in `jp_class.py`.

### Adding or Updating Data

To add new data files:

1. Create the appropriate directory under `Data_for_class/`
2. Add data files following the expected format for that dataset type
3. Update the `catalog` dictionary in `jp_class.py` with the new filepath

Example:
```python
'MyDataset': {
    'data_filepath': 'Data_for_class/MyDataset',
    'load_fn': my_data_function,
    'theory_fn': my_theory_function,
    'emulators': ['cp_DA']
}
```

### Adding or Updating Emulators

To add new emulator files:

1. Place emulator `.npz` files in `Emulators/<parameter_space>/`
2. Update the `emulator_config` dictionary in `jp_class.py` if adding new emulator types

Example:
```python
'cp_new': {
    'probe': 'custom_log',
    'filepath': '/new_emulator.npz'
}
```

## Usage Example

```python
from pipeline.jp_class import stats_quantities

# Initialize with datasets and parameter space
requested_data = ['DES_Y5']
extensions = 'Base'

stats = stats_quantities(requested_data, extensions)

# Access different statistical measures
params_dict = {'ombh2': 0.022, 'omch2': 0.12, 'h': 0.67, 'ns': 0.96, 
               'logA': 3.05, 'tau': 0.06, 'w0': -1.0, 'wa': 0.0, 'M_b': -19.3}

log_like = stats.log_likelihood(params_dict)
chi2 = stats.chi_square(params_dict)
```

## Working Directory

When running scripts that use `jp_class.py`, ensure you're in the repository root directory:

```bash
cd /path/to/Jeffreys-prior-project
python your_script.py
```

Or add the repository to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/Jeffreys-prior-project')
from pipeline.jp_class import stats_quantities
```

## Common Setup Issues

### Issue: Module import errors
**Solution**: Ensure you're in the repository root directory when running scripts, or add it to `PYTHONPATH`:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Jeffreys-prior-project"
```

### Issue: FileNotFoundError for data files
**Solution**: Verify the file structure matches the expected layout. Check that all data files are present in `Data_for_class/` subdirectories.

To debug file paths:
```python
import os
print(os.getcwd())  # Verify current working directory
print(os.path.exists('Data_for_class/DES_Y5/DES-SN5YR_HD.csv'))  # Check if file exists
```

### Issue: JAX not finding GPU
**Solution**: Install the appropriate JAX version for your hardware:
```bash
# For CUDA 11.8
pip install jax[cuda11_cudnn82]

# For CPU only
pip install jax
```

### Issue: CosmoPower emulator loading fails
**Solution**: Ensure emulator `.npz` files are in the correct directory structure under `Emulators/<parameter_space>/`.

## Testing Your Setup

Run this quick test to verify everything is configured correctly:

```python
from pipeline.jp_class import stats_quantities

try:
    stats = stats_quantities(['DES_Y5'], 'Base')
    print("✓ Successfully loaded DES_Y5 dataset")
    print("✓ Successfully loaded Base emulators")
    print("\nSetup is working correctly!")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Check file paths and dependencies")
```

## Next Steps

1. Review the [Project Overview](PROJECT_OVERVIEW.md) for scientific context
2. Examine `pipeline/jp_class.py` for available methods and their usage
3. Check the `Emulators/` directory structure to understand available parameter spaces
4. Review `Data_for_class/` to see what datasets are available
5. Run analysis scripts from the repository root directory

## Data File Requirements

Each dataset type requires specific files in its directory. Refer to the `load_*` functions in `pipeline/jp_class.py` for exact file format requirements:

- **BAO data**: `data.txt`, `cov_mat.txt`, `model_order.txt`
- **Supernovae data**: CSV/DAT file with redshift and magnitude columns, covariance matrix
- **CMB data**: Fortran binary files and ASCII text files with power spectrum information
- **Planck low-ℓ data**: Log-normal distribution parameters and bin information

