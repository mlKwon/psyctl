# End-to-End Test Script for PSYCTL (BiPO Method)
# This script performs a complete pipeline test: dataset build -> extract (BiPO) -> steering

# Set UTF-8 encoding for PowerShell output
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Load environment variables from .env file
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)\s*=\s*(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            Set-Item -Path "env:$name" -Value $value
            Write-Host "Loaded: $name" -ForegroundColor Green
        }
    }
} else {
    Write-Host "Warning: .env file not found" -ForegroundColor Yellow
}

# Clear results directory
$resultsDir = "./results/test_e2e_bipo"
if (Test-Path $resultsDir) {
    Write-Host "Cleaning $resultsDir..." -ForegroundColor Cyan
    Remove-Item -Path $resultsDir -Recurse -Force
}
New-Item -Path $resultsDir -ItemType Directory -Force | Out-Null

# Step 1: Build steering dataset
Write-Host "`n=== Step 1: Building Steering Dataset ===" -ForegroundColor Cyan
uv run psyctl dataset.build.steer `
    --model "google/gemma-3-270m-it" `
    --personality "Rejection" `
    --limit-samples 100 `
    --output "./results/test_e2e_bipo/reject_steer_dataset"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Dataset build failed" -ForegroundColor Red
    exit 1
}

# Step 2: Extract steering vector using BiPO
Write-Host "`n=== Step 2: Extracting Steering Vector (BiPO) ===" -ForegroundColor Cyan
uv run psyctl extract.steering `
    --model "google/gemma-3-270m-it" `
    --layer "model.layers[13].mlp" `
    --method bipo `
    --lr 5e-4 `
    --beta 0.1 `
    --epochs 5 `
    --dataset "./results/test_e2e_bipo/reject_steer_dataset" `
    --output "./results/test_e2e_bipo/reject_steer_bipo.safetensors"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Steering vector extraction failed" -ForegroundColor Red
    exit 1
}

# Step 3: Apply steering
Write-Host "`n=== Step 3: Applying Steering ===" -ForegroundColor Cyan
uv run psyctl steering `
    --model "google/gemma-3-270m-it" `
    --steering-vector "./results/test_e2e_bipo/reject_steer_bipo.safetensors" `
    --input-text "Tell me about yourself"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Steering application failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== E2E Test (BiPO) Completed Successfully ===" -ForegroundColor Green
