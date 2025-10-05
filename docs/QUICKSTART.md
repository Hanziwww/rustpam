# Quick Start Guide

## Installation

### Method 1: Build from Source (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd rustpam

# 2. Install Rust (if not already installed)
# Visit https://rustup.rs/

# 3. Install maturin
pip install maturin

# 4. Build and install (Windows PowerShell)
.\build_and_install.ps1

# Or on Linux/macOS
./build_and_install.sh

# Or manually execute
maturin build --release
pip install target/wheels/rustpam-*.whl
```

### Method 2: Direct Install from Wheel (if pre-built available)

```bash
pip install rustpam-*.whl
```

## Verify Installation

```bash
python test_basic.py
```

If you see "All tests passed! ✓", the installation was successful!

## Basic Usage

### Example 1: Simple Clustering

```python
import numpy as np
from rustpam import OneBatchPAM

# Generate test data
np.random.seed(42)
X = np.random.randn(100, 5).astype(np.float32)

# Create model and fit
model = OneBatchPAM(n_medoids=3, random_state=42)
model.fit(X)

# View results
print("Medoid indices:", model.medoid_indices_)
print("Cluster labels:", model.labels_[:10])  # Labels of first 10 samples
print("Inertia:", model.inertia_)
```

### Example 2: Custom Parameters

```python
model = OneBatchPAM(
    n_medoids=5,              # Number of medoids
    distance='manhattan',     # Distance metric: 'euclidean' or 'manhattan'
    max_iter=100,            # Maximum iterations
    random_state=42,         # Random seed
    n_threads=4              # Number of parallel threads
)
model.fit(X)
```

### Example 3: Predict New Data

```python
# Train model
model = OneBatchPAM(n_medoids=3)
model.fit(X_train)

# Predict new samples
X_test = np.random.randn(20, 5).astype(np.float32)
labels = model.predict(X_test)
print("Test set labels:", labels)
```

### Example 4: Get Medoids

```python
model.fit(X)

# Get medoid indices
medoid_indices = model.medoid_indices_
print("Medoid indices:", medoid_indices)

# Get medoid vectors
medoids = X[medoid_indices]
print("Medoid shape:", medoids.shape)
```

## Parameter Documentation

### OneBatchPAM

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_medoids` | int | Required | Number of clusters (number of medoids) |
| `distance` | str | 'euclidean' | Distance metric: 'euclidean' or 'manhattan' |
| `max_iter` | int | 50 | Maximum number of iterations |
| `random_state` | int/None | None | Random seed for reproducible results |
| `n_threads` | int | 1 | Number of parallel threads (0 = auto-detect) |

### Attributes

- `medoid_indices_`: Array of medoid indices
- `labels_`: Cluster label for each sample
- `inertia_`: Total distance from samples to nearest medoid
- `n_iter_`: Actual number of iterations

## Performance Tips

1. **Data Type**: Use `float32` instead of `float64` for significant performance gains
   ```python
   X = X.astype(np.float32)
   ```

2. **Parallelization**: Set appropriate number of threads
   ```python
   import os
   n_cores = os.cpu_count()
   model = OneBatchPAM(n_threads=n_cores)
   ```

3. **Distance Metric**: Manhattan distance is typically faster than Euclidean
   ```python
   model = OneBatchPAM(distance='manhattan')
   ```

4. **Compilation Optimization**: Ensure building with `--release` mode
   ```bash
   maturin build --release
   ```

## Comparison with scikit-learn

RustPAM's API is designed to be compatible with scikit-learn:

```python
# scikit-learn KMeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(X)

# RustPAM OneBatchPAM (similar API)
from rustpam import OneBatchPAM
pam = OneBatchPAM(n_medoids=3, random_state=42)
pam.fit(X)
```

Key differences:
- KMeans uses centroids, PAM uses medoids (actual data points)
- PAM is more robust to outliers
- RustPAM is implemented in Rust for excellent performance

## Troubleshooting

### ImportError: cannot import name 'swap_eager'

Ensure the package is installed correctly:
```bash
pip uninstall rustpam
maturin build --release
pip install target/wheels/rustpam-*.whl --force-reinstall
```

### Performance Not as Expected

1. Ensure building in release mode
2. Check data type is `float32`
3. Increase `n_threads` parameter
4. Consider using 'manhattan' distance

### Windows Encoding Issues

If you see garbled text, add at the beginning of your script:
```python
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## More Examples

See the following files for more examples:
- `test_basic.py` - Basic functionality tests
- `README.md` - Detailed documentation
- `DEVELOP.md` - Development guide

## Getting Help

If you have questions:
1. Check `README.md` for detailed documentation
2. Run `python test_basic.py` to verify installation
3. Submit an Issue to the project repository
