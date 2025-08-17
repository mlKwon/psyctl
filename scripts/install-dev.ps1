# 개발 환경 설치 스크립트
Write-Host "Installing development environment..." -ForegroundColor Green

# uv 설치 확인
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
    & .\install_uv.ps1
    Remove-Item "install_uv.ps1"
}

# 프로젝트 초기화
Write-Host "Initializing project..." -ForegroundColor Yellow
uv venv
& .\.venv\Scripts\Activate.ps1

# 의존성 설치
Write-Host "Installing dependencies..." -ForegroundColor Yellow
uv sync

Write-Host "Development environment setup complete!" -ForegroundColor Green
