.PHONY: help install dev test lint format clean build release pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in production mode"
	@echo "  dev          - Install package in development mode"
	@echo "  test         - Run tests"
	@echo "  lint         - Run all linters (Rust + Python)"
	@echo "  format       - Format code (Rust + Python)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build release version"
	@echo "  release      - Build release wheels"
	@echo "  pre-commit   - Install pre-commit hooks"
	@echo "  check        - Run format checks and linters"

# Install package for production
install:
	pip install --upgrade pip
	maturin build --release
	pip install target/wheels/*.whl --force-reinstall

# Install package for development
dev:
	pip install --upgrade pip
	pip install maturin
	pip install -e . --no-build-isolation
	maturin develop

# Run tests
test:
	pytest test/ -v --cov=rustpam --cov-report=term-missing

# Run all linters
lint: lint-rust lint-python

# Lint Rust code
lint-rust:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

# Lint Python code
lint-python:
	black --check rustpam/ test/
	isort --check-only rustpam/ test/
	flake8 rustpam/ test/
	mypy rustpam/ || true

# Format all code
format: format-rust format-python

# Format Rust code
format-rust:
	cargo fmt --all

# Format Python code
format-python:
	black rustpam/ test/
	isort rustpam/ test/

# Clean build artifacts
clean:
	rm -rf target/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dll" -delete

# Build release version
build:
	maturin build --release

# Build release wheels for distribution
release:
	maturin build --release --strip

# Install pre-commit hooks
pre-commit:
	pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

# Run checks (format + lint)
check: lint
	@echo "✓ All checks passed!"

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files

# Development environment setup
setup: dev pre-commit
	pip install pytest pytest-cov pytest-xdist pytest-timeout
	pip install black isort flake8 mypy
	@echo "✓ Development environment setup complete!"

