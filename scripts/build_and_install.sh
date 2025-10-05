#!/bin/bash
# Shell Script - Build and Install RustPAM

echo "========================================"
echo "  Building and Installing RustPAM"
echo "========================================"
echo ""

# Build project
echo "Step 1: Building Rust project..."
maturin build --release
if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi
echo "✓ Build successful!"
echo ""

# Find generated wheel file
WHEEL_FILE=$(ls -t target/wheels/*.whl 2>/dev/null | head -n1)

if [ -n "$WHEEL_FILE" ]; then
    echo "Step 2: Installing wheel package..."
    echo "Wheel file: $(basename $WHEEL_FILE)"
    
    # Uninstall old version
    pip uninstall -y rustpam 2>/dev/null
    
    # Install new version
    pip install "$WHEEL_FILE" --force-reinstall
    
    if [ $? -eq 0 ]; then
        echo "✓ Installation successful!"
    else
        echo "✗ Installation failed!"
        exit 1
    fi
else
    echo "✗ Wheel file not found!"
    exit 1
fi

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Run tests:"
echo "  python test_basic.py"
echo ""
echo "Run performance comparison:"
echo "  python compare_performance.py"
echo ""
