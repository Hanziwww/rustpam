# RustPAM

High-performance k-medoids (OneBatchPAM) clustering with a Rust core.

[![PyPI version](https://badge.fury.io/py/rustpam.svg)](https://badge.fury.io/py/rustpam)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Installation

Install from PyPI:

```bash
pip install rustpam
```

For development installation:

```bash
# Clone the repository
git clone https://github.com/yourusername/rustpam.git
cd rustpam

# Install in development mode
pip install maturin
maturin develop --release
```

## Quick Start

```python
import numpy as np
from rustpam import OneBatchPAM

# Sample data
X = np.random.randn(1000, 10).astype(np.float32)

# Create and fit model
model = OneBatchPAM(
    n_medoids=3,
    distance="euclidean",
    max_iter=50,
    random_state=42,
    n_threads=2,
)
model.fit(X)

# Results
print("Medoid indices:", model.medoid_indices_)
print("Inertia:", model.inertia_)
print("Iterations:", model.n_iter_)

# Predict new samples
X_new = np.random.randn(20, X.shape[1]).astype(np.float32)
labels = model.predict(X_new)
print("Predicted labels:", labels)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [Maturin](https://www.maturin.rs/) - Build tool for Rust Python packages
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism library
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays
