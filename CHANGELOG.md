# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2024-10-05

### Fixed
- **CRITICAL**: Fixed `fit_predict()` method to return labels array (shape: n_samples,) instead of medoid_indices array (shape: n_medoids,)
- Updated `fit()` method to compute full-sample labels and distances to nearest medoids for better accuracy
- Improved documentation for `fit_predict()` method to clarify return type

### Changed
- `fit()` now computes labels and distances using full pairwise distances to cluster centers instead of using batch solution

## [0.2.0] - 2024-10-05

### Added
- Initial release of rustpam
- OneBatchPAM implementation with Rust backend
- Python bindings using PyO3
- Support for custom distance metrics
- Parallel distance computation using Rayon
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Documentation and examples

## [0.1.0] - 2024-10-05

### Added
- Initial public release
- Core k-medoids clustering functionality
- Python 3.10-3.13 support
- Cross-platform support (Linux, Windows, macOS)
- scikit-learn compatible API
- Parallel processing support
- Comprehensive documentation

[Unreleased]: https://github.com/Hanziwww/rustpam/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Hanziwww/rustpam/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Hanziwww/rustpam/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Hanziwww/rustpam/releases/tag/v0.1.0
