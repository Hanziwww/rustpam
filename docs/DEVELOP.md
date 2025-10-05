# Development Guide

This document describes how to develop and build the RustPAM project.

## Requirements

### Essential Tools
- **Rust**: >= 1.70 (install: https://rustup.rs/)
- **Python**: >= 3.8
- **maturin**: Python package build tool

### Install maturin
```bash
pip install maturin
```

## Project Structure

```
rustpam/
├── src/
│   └── lib.rs              # Rust core implementation
├── rustpam/
│   ├── __init__.py         # Python package initialization
│   └── onebatchpam.py      # Python wrapper layer
├── Cargo.toml              # Rust dependency configuration
├── pyproject.toml          # Python project configuration
├── README.md               # Project documentation
├── QUICKSTART.md           # Quick start guide
├── DEVELOP.md              # Development guide (this file)
├── test_basic.py           # Basic tests
└── compare_performance.py  # Performance comparison
```

## Development Workflow

### 1. Clone Project
```bash
git clone <repository-url>
cd rustpam
```

### 2. Check Rust Code
```bash
cargo check
```

### 3. Run Rust Tests
```bash
cargo test
```

### 4. Build Project

#### Debug Mode (fast iteration)
```bash
cargo build
maturin build
```

#### Release Mode (performance optimized)
```bash
cargo build --release
maturin build --release
```

### 5. Development Mode Installation

Create virtual environment and install:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install
maturin develop --release
```

Or directly install wheel:
```bash
# Windows
.\build_and_install.ps1

# Linux/macOS
./build_and_install.sh
```

### 6. Run Tests
```bash
python test_basic.py
python compare_performance.py
```

## Code Modification Workflow

### Modifying Rust Code
1. Edit `src/lib.rs`
2. Run `cargo check` to check syntax
3. Run `cargo build --release` to build
4. Run `maturin build --release` to generate wheel
5. Install and test

### Modifying Python Code
1. Edit `rustpam/*.py`
2. If only Python code is modified, no need to rebuild Rust
3. Run tests directly to verify

## Performance Optimization

### Rust Optimization Options

Configure optimization level in `Cargo.toml`:

```toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = true            # Link-time optimization
codegen-units = 1     # Better optimization, but slower compilation
```

### Parallelization Tuning

Adjust thread count for optimal performance:

```python
model = OneBatchPAM(
    n_threads=4,  # Adjust based on CPU core count
    ...
)
```

## Debugging

### Rust Debugging
```bash
# Use println! or dbg! macros
cargo build && cargo test -- --nocapture
```

### Python Debugging
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IDE debugger
```

## Performance Profiling

### Rust profiling
```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate flame graph
cargo flamegraph
```

### Python profiling
```python
import cProfile
cProfile.run('model.fit(X)')
```

## Code Standards

### Rust
- Use `rustfmt` to format code
  ```bash
  cargo fmt
  ```
- Use `clippy` to check code quality
  ```bash
  cargo clippy
  ```

### Python
- Follow PEP 8
- Use `black` for formatting
  ```bash
  pip install black
  black rustpam/
  ```

## Common Issues

### Q: maturin cannot find virtual environment
**A:** Use `maturin build` instead of `maturin develop`, then manually install wheel:
```bash
pip install target/wheels/*.whl
```

### Q: Version conflict errors
**A:** Ensure PyO3 and numpy versions in `Cargo.toml` are compatible:
```toml
pyo3 = "0.23.3"
numpy = "0.23"
```

### Q: Build fails on Windows
**A:** Ensure Visual Studio Build Tools or MinGW is installed

### Q: Performance not as expected
**A:** 
1. Ensure building with `--release` mode
2. Adjust `n_threads` parameter
3. Check data is `float32` type

## Contributing Guidelines

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Release Process

1. Update version numbers
   - `Cargo.toml`: `version = "x.y.z"`
   - `rustpam/__init__.py`: `__version__ = "x.y.z"`

2. Build wheel
   ```bash
   maturin build --release
   ```

3. Upload to PyPI
   ```bash
   pip install twine
   twine upload target/wheels/*
   ```

## References

- [PyO3 Documentation](https://pyo3.rs/)
- [maturin Documentation](https://www.maturin.rs/)
- [Rayon Documentation](https://docs.rs/rayon/)
- [ndarray Documentation](https://docs.rs/ndarray/)
