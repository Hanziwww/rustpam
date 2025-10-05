.PHONY: help install dev test format lint clean build release

help:
	@echo "RustPAM Development Commands"
	@echo "============================"
	@echo "install      - Install package in release mode"
	@echo "dev          - Install package in development mode"
	@echo "test         - Run tests"
	@echo "format       - Format Rust and Python code"
	@echo "lint         - Run linters on Rust and Python code"
	@echo "clean        - Clean build artifacts"
	@echo "build        - Build release wheels"
	@echo "build-dev    - Build development wheel"
	@echo "release      - Create a new release (use VERSION=x.y.z)"

install:
	pip install maturin
	maturin build --release
	pip install --force-reinstall target/wheels/*.whl

dev:
	pip install maturin numpy scikit-learn pytest pytest-cov
	maturin develop --release

test:
	pytest test/ -v

format:
	@echo "Formatting Rust code..."
	cargo fmt --all
	@echo "Formatting Python code..."
	black rustpam/ test/
	isort rustpam/ test/

lint:
	@echo "Linting Rust code..."
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	@echo "Linting Python code..."
	black --check rustpam/ test/
	isort --check-only rustpam/ test/
	flake8 rustpam/ test/
	mypy rustpam/ || true

clean:
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete

build:
	maturin build --release

build-dev:
	maturin build

release:
ifndef VERSION
	@echo "Error: VERSION not specified. Usage: make release VERSION=x.y.z"
	@exit 1
endif
	@echo "Creating release $(VERSION)..."
	python scripts/bump_version.py $(VERSION)
	@echo ""
	@echo "Next steps:"
	@echo "1. Review changes: git diff"
	@echo "2. Update CHANGELOG.md"
	@echo "3. Commit: git add -A && git commit -m 'Bump version to $(VERSION)'"
	@echo "4. Tag: git tag -a v$(VERSION) -m 'Release version $(VERSION)'"
	@echo "5. Push: git push origin main && git push origin v$(VERSION)"

