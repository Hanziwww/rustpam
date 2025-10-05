# RustPAM

High-performance k-medoids (PAM) clustering with a Rust core, exposed to Python via PyO3. Parallelized with Rayon, offering strong performance and a scikit-learn–style API.

[![PyPI version](https://badge.fury.io/py/rustpam.svg)](https://badge.fury.io/py/rustpam)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **High Performance**: Rust-powered core with parallel processing
- 🐍 **Python Integration**: scikit-learn compatible API
- 🔧 **Flexible**: Support for custom distance metrics
- 🌐 **Cross-Platform**: Linux, Windows, macOS (x86_64 and ARM64)
- 📦 **Easy Install**: Pre-built wheels for Python 3.10-3.13

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

## Documentation

- [Development Guide](docs/DEVELOP.md) - Setup and development workflow
- [Release Guide](docs/RELEASE.md) - How to release new versions
- [API Documentation](https://github.com/yourusername/rustpam#api) - Coming soon

## Performance

RustPAM leverages Rust's performance and Rayon's parallel processing to deliver fast clustering:

- Parallel distance computations
- Efficient memory usage
- Optimized swap operations
- Native code performance

## Contributing

Contributions are welcome! Please see [DEVELOP.md](docs/DEVELOP.md) for development setup and guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [Maturin](https://www.maturin.rs/) - Build tool for Rust Python packages
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism library
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays
