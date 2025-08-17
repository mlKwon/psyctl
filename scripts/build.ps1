# 빌드 스크립트
Write-Host "Building psyctl..." -ForegroundColor Green

# 가상환경 활성화
& .\.venv\Scripts\Activate.ps1

# 코드 포맷팅
Write-Host "Formatting code..." -ForegroundColor Yellow
uv run black src/
uv run isort src/

# 린팅
Write-Host "Running linters..." -ForegroundColor Yellow
uv run flake8 src/
uv run mypy src/

# 테스트
Write-Host "Running tests..." -ForegroundColor Yellow
uv run pytest

# 설치
Write-Host "Installing package..." -ForegroundColor Yellow
uv pip install -e .

Write-Host "Build complete!" -ForegroundColor Green
