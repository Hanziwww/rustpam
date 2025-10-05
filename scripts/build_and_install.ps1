# PowerShell Script - Build and Install RustPAM

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Building and Installing RustPAM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Build project
Write-Host "Step 1: Building Rust project..." -ForegroundColor Yellow
maturin build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build successful!" -ForegroundColor Green
Write-Host ""

# Find generated wheel file
$wheelFile = Get-ChildItem -Path "target\wheels\*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($wheelFile) {
    Write-Host "Step 2: Installing wheel package..." -ForegroundColor Yellow
    Write-Host "Wheel file: $($wheelFile.Name)" -ForegroundColor Gray
    
    # Uninstall old version
    pip uninstall -y rustpam 2>$null
    
    # Install new version
    pip install $wheelFile.FullName --force-reinstall
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Installation successful!" -ForegroundColor Green
    } else {
        Write-Host "✗ Installation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ Wheel file not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run tests:" -ForegroundColor Yellow
Write-Host "  python test_basic.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Run performance comparison:" -ForegroundColor Yellow
Write-Host "  python compare_performance.py" -ForegroundColor Gray
Write-Host ""
