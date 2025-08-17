# 코드 포맷팅 스크립트
Write-Host "Formatting code..." -ForegroundColor Green

# 가상환경 활성화
& .\.venv\Scripts\Activate.ps1

# 포맷팅 실행
uv run black src/
uv run isort src/

Write-Host "Code formatting complete!" -ForegroundColor Green
