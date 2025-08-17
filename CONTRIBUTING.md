# 개발자 가이드

PSYCTL 프로젝트에 기여하고 싶으시다면 환영합니다! 이 문서는 개발 환경 설정과 기여 방법을 안내합니다.

## 📋 목차

- [개발 환경 설정](#개발-환경-설정)
- [프로젝트 구조](#프로젝트-구조)
- [개발 워크플로우](#개발-워크플로우)
- [코딩 스타일](#코딩-스타일)
- [테스트](#테스트)
- [기여 방법](#기여-방법)

## 🛠️ 개발 환경 설정

### 필수 요구사항

- Python 3.9+
- uv (패키지 관리자)
- Git

### 초기 설정

```bash
# 1. 저장소 클론
git clone https://github.com/modulabs-personalab/psyctl.git
cd psyctl

# 2. 개발 환경 설정 (Windows)
& .\scripts\install-dev.ps1

# 3. 가상환경 활성화
& .\.venv\Scripts\Activate.ps1
```

### 수동 설정

```bash
# uv 설치
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# 가상환경 생성
uv venv

# 의존성 설치
uv sync

# 개발 의존성 설치
uv add --dev pytest black isort flake8 mypy
```

## 📁 프로젝트 구조

```
psyctl/
├── pyproject.toml              # 프로젝트 설정 및 의존성
├── README.md                   # 사용자 가이드
├── CONTRIBUTING.md             # 개발자 가이드 (이 파일)
├── .gitignore                  # Git 무시 파일
├── src/                        # 소스 코드
│   └── psyctl/                 # 메인 패키지
│       ├── __init__.py
│       ├── cli.py              # CLI 진입점
│       ├── commands/           # 명령어 모듈들
│       │   ├── dataset.py      # 데이터셋 생성
│       │   ├── extract.py      # 스티어링 벡터 추출
│       │   ├── steering.py     # 스티어링 실험
│       │   └── benchmark.py    # 인벤토리 테스트
│       ├── core/               # 핵심 로직
│       │   ├── dataset_builder.py
│       │   ├── steering_extractor.py
│       │   ├── steering_applier.py
│       │   ├── inventory_tester.py
│       │   ├── utils.py
│       │   └── logger.py       # 로깅 설정
│       ├── models/             # 모델 관련
│       │   ├── llm_loader.py
│       │   └── vector_store.py
│       ├── data/               # 데이터 관련
│       │   ├── personality_templates.py
│       │   └── inventories/    # 인벤토리 데이터
│       └── config/             # 설정 관리
│           └── settings.py
├── tests/                      # 테스트 코드
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_commands/
├── scripts/                    # 개발 스크립트
│   ├── install-dev.ps1
│   ├── build.ps1
│   ├── test.ps1
│   └── format.ps1
└── docs/                       # 문서
    └── README.md
```

## 🔄 개발 워크플로우

### 1. 브랜치 생성

```bash
# 메인 브랜치에서 새 브랜치 생성
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. 개발 및 테스트

```bash
# 코드 포맷팅
& .\scripts\format.ps1

# 테스트 실행
& .\scripts\test.ps1

# 빌드 확인
& .\scripts\build.ps1
```

### 3. 커밋 및 푸시

```bash
# 변경사항 스테이징
git add .

# 커밋
git commit -m "feat: add new feature description"

# 푸시
git push origin feature/your-feature-name
```

### 4. Pull Request 생성

GitHub에서 Pull Request를 생성하고 다음을 포함하세요:
- 변경사항 설명
- 테스트 결과
- 관련 이슈 번호

## 📝 코딩 스타일

### Python 코드 스타일

- **Black**: 코드 포맷팅
- **isort**: import 정렬
- **flake8**: 린팅
- **mypy**: 타입 체크

### 명명 규칙

- **클래스**: PascalCase (`DatasetBuilder`)
- **함수/변수**: snake_case (`build_caa_dataset`)
- **상수**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`)
- **모듈**: snake_case (`dataset_builder.py`)

### 문서화

- 모든 공개 함수와 클래스에 docstring 작성
- Google 스타일 docstring 사용
- 타입 힌트 사용

```python
def build_caa_dataset(self, model: str, personality: str, output_dir: Path) -> None:
    """Build CAA dataset for given personality traits.
    
    Args:
        model: Model name to use for dataset generation
        personality: Comma-separated personality traits
        output_dir: Directory to save the dataset
        
    Raises:
        FileNotFoundError: If model cannot be loaded
        ValueError: If personality traits are invalid
    """
    pass
```

## 🧪 테스트

### 테스트 실행

```bash
# 모든 테스트 실행
uv run pytest

# 특정 테스트 실행
uv run pytest tests/test_cli.py

# 커버리지와 함께 실행
uv run pytest --cov=psyctl --cov-report=html
```

### 테스트 작성 가이드

- 테스트 파일명: `test_*.py`
- 테스트 함수명: `test_*`
- 각 테스트는 독립적이어야 함
- Mock 사용하여 외부 의존성 격리

```python
def test_build_caa_dataset():
    """Test CAA dataset building functionality."""
    # Arrange
    builder = DatasetBuilder()
    
    # Act
    result = builder.build_caa_dataset("test-model", "Extroversion", Path("./test"))
    
    # Assert
    assert result is not None
```

## 🤝 기여 방법

### 이슈 리포트

버그 리포트나 기능 요청 시 다음을 포함하세요:
- 문제/요청 설명
- 재현 단계
- 예상 동작
- 실제 동작
- 환경 정보 (OS, Python 버전 등)

### 기능 개발

1. **이슈 생성**: 개발할 기능에 대한 이슈 생성
2. **브랜치 생성**: `feature/issue-number-description` 형식
3. **개발**: 기능 구현 및 테스트 작성
4. **테스트**: 모든 테스트 통과 확인
5. **문서화**: README나 API 문서 업데이트
6. **PR 생성**: Pull Request 생성

### 버그 수정

1. **이슈 확인**: 기존 이슈가 있는지 확인
2. **브랜치 생성**: `fix/issue-number-description` 형식
3. **수정**: 버그 수정 및 테스트 추가
4. **검증**: 수정 사항이 다른 기능에 영향을 주지 않는지 확인
5. **PR 생성**: Pull Request 생성

## 📋 체크리스트

PR 제출 전 다음 사항을 확인하세요:

- [ ] 코드가 코딩 스타일을 준수하는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 새로운 기능에 대한 테스트가 작성되었는가?
- [ ] 문서가 업데이트되었는가?
- [ ] 커밋 메시지가 명확한가?
- [ ] PR 설명이 충분한가?

## 🚀 릴리스 프로세스

### 버전 관리

- **Semantic Versioning** 사용 (MAJOR.MINOR.PATCH)
- `pyproject.toml`의 `version` 필드 업데이트
- 변경사항을 `CHANGELOG.md`에 기록

### 릴리스 단계

1. **개발**: `main` 브랜치에서 개발
2. **테스트**: 모든 테스트 통과 확인
3. **버전 업데이트**: `pyproject.toml` 버전 수정
4. **태그 생성**: `git tag v1.0.0`
5. **배포**: GitHub Releases에 업로드

## 📞 문의

개발 관련 문의사항이 있으시면:
- GitHub Issues 사용
- 프로젝트 메인테이너에게 직접 연락

## 📄 라이센스

기여하신 코드는 MIT 라이센스 하에 배포됩니다.
