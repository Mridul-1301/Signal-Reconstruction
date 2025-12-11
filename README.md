# ECG Signal Imputation Operator - Clean Implementation

This is a clean, streamlined version of the ECG signal imputation operator with implicit gradient training.

## Directory Structure

- **`data/`** - Contains the ECG data file (`ecg_missing.txt`)
- **`src/`** - Clean implementation of the core components:
  - `config.py` - Configuration classes
  - `ops.py` - Linear operators (D, D^T, A, A^T)
  - `weights.py` - Safe weight parameterization and neural network
  - `solver.py` - Conjugate gradient solver
  - `layer.py` - Main autograd module with implicit gradient support
- **`main/`** - Demo script (`demo_ecg_operator.py`)

## Key Features

- **Implicit Mode**: Adjoint gradient through optimizer for general loss functions
- **Safe Weight Parameterization**: Prevents weight collapse with mean scale normalization
- **Conjugate Gradient Solver**: Efficient solution of normal equations
- **Linear Operators**: Proper adjoint implementation for numerical stability

## Usage

Run the demo script:

```bash
cd main
python demo_ecg_operator.py
```

This will demonstrate the implicit gradient method on real ECG data with missing values.

## Improvements Made

1. **Removed duplicate code** - Consolidated repeated functionality
2. **Cleaner interfaces** - Simplified function signatures and removed unused parameters
3. **Better organization** - Logical separation of concerns
4. **Streamlined demo** - Single script with shared utilities
5. **Removed test code** - Kept only essential functionality
6. **Improved documentation** - Clear docstrings and comments
