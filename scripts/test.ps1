# 테스트 실행 스크립트
Write-Host "Running tests..." -ForegroundColor Green

# 가상환경 활성화
& .\.venv\Scripts\Activate.ps1

# 테스트 실행
uv run pytest -v --cov=psyctl --cov-report=html

Write-Host "Tests complete! Coverage report generated in htmlcov/" -ForegroundColor Green
