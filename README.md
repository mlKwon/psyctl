# PSYCTL - LLM Personality Steering Tool

모두의 연구소 [페르소나랩](https://modulabs.co.kr/labs/337) 에서 진행하는 프로젝트 입니다.

LLM을 지정된 성격으로 steering 하는 것을 지원하는 툴입니다. 자동으로 데이터셋을 생성하여 모델과 성격만 지정하면 작동하는 것이 목표입니다.

---

## 📖 사용자 가이드

### 🚀 빠른 시작

#### 설치

**기본 설치 (CPU 버전)**
```bash
# uv 설치 (Windows)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# 프로젝트 설정
uv venv
& .\.venv\Scripts\Activate.ps1
uv sync
```

**GPU 가속 설치 (CUDA 지원)**
```bash
# 기본 설치 후 CUDA 지원 PyTorch 설치
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **중요**: `transformers` 패키지가 `torch`를 의존성으로 가지고 있어서, `uv sync`를 실행하면 자동으로 CPU 버전이 설치됩니다. GPU 사용을 위해서는 위의 CUDA 설치 명령어를 다시 실행해야 합니다.

#### 기본 사용법

```bash
# 1. 데이터셋 생성
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/cca"

# 2. 스티어링 벡터 추출
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/cca" \
  --output "./steering_vector/out.safetensors"

# 3. 스티어링 실험
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello world blabla"

# 4. 인벤토리 테스트
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory IPIP-NEO
```

### 📋 명령어 상세 가이드

#### 1. 데이터셋 생성 (`dataset.build.caa`)

스티어링 벡터를 찾기 위한 데이터셋을 생성합니다.

```bash
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/cca"
```

**매개변수:**
- `--model`: 사용할 모델명 (Hugging Face 모델 ID)
- `--personality`: 대상 성격 특성 (쉼표로 구분)
- `--output`: 데이터셋 저장 경로

#### 2. 스티어링 벡터 추출 (`extract.steering`)

CAA 방법을 사용하여 스티어링 벡터를 추출합니다.

```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/cca" \
  --output "./steering_vector/out.safetensors"
```

**매개변수:**
- `--model`: 사용할 모델명
- `--layer`: 활성화를 추출할 레이어 경로
- `--dataset`: 데이터셋 경로
- `--output`: 스티어링 벡터 저장 경로 (.safetensors)

#### 3. 스티어링 실험 (`steering`)

추출된 스티어링 벡터를 적용하여 텍스트를 생성합니다.

```bash
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello world blabla"
```

**매개변수:**
- `--model`: 사용할 모델명
- `--steering-vector`: 스티어링 벡터 파일 경로
- `--input-text`: 입력 텍스트

#### 4. 인벤토리 테스트 (`benchmark`)

심리학적 인벤토리를 사용하여 성격 변화를 측정합니다.

```bash
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory IPIP-NEO
```

**매개변수:**
- `--model`: 사용할 모델명
- `--steering-vector`: 스티어링 벡터 파일 경로
- `--inventory`: 사용할 인벤토리명

### 📊 지원하는 인벤토리

| Inventory | Domain | License | Notes |
|-----------|--------|---------|-------|
| IPIP-NEO-300/120 | Big Five | Public Domain | Full & short forms |
| NPI-40 | Narcissism | Free research use | Forced-choice |
| PNI-52 | Pathological narcissism | CC-BY-SA | Likert 1–6 |
| NARQ-18 | Admiration & Rivalry | CC-BY-NC | Two sub-scales |
| MACH-IV | Machiavellianism | Public Domain | Likert 1–5 |
| LSRP-26 | Psychopathy | Public Domain | Primary & secondary |
| PPI-56 | Psychopathy | Free research use | Short form |

### ⚙️ 설정

#### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다:

```bash
# .env 파일 예시
PSYCTL_LOG_LEVEL=INFO
HF_TOKEN=your_huggingface_token_here
```

#### 로그 레벨 설정

환경 변수나 `.env` 파일을 통해 로그 레벨을 설정할 수 있습니다:

```bash
PSYCTL_LOG_LEVEL=DEBUG
```

#### Hugging Face 토큰 설정

일부 모델에 접근하려면 Hugging Face 토큰이 필요합니다:

1. [Hugging Face 설정 페이지](https://huggingface.co/settings/tokens)에서 토큰을 생성
2. `.env` 파일에 `HF_TOKEN=your_token_here` 추가
3. 또는 환경 변수로 설정: `export HF_TOKEN=your_token_here`

#### 출력 디렉토리

기본적으로 다음 디렉토리들이 자동으로 생성됩니다:
- `./dataset/` - 데이터셋 저장
- `./steering_vector/` - 스티어링 벡터 저장
- `./results/` - 결과 저장
- `./output/` - 기타 출력 파일

### 📝 예시

#### 전체 워크플로우 예시

```bash
# 1. 외향성 성격을 위한 데이터셋 생성
psyctl dataset.build.caa \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Extroversion" \
  --output "./dataset/extroversion"

# 2. 스티어링 벡터 추출
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/extroversion" \
  --output "./steering_vector/extroversion.safetensors"

# 3. 스티어링 적용하여 텍스트 생성
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --input-text "Tell me about yourself"

# 4. 성격 변화 측정
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --inventory IPIP-NEO
```

#### Python 라이브러리로 사용하기

PSYCTL은 CLI 도구뿐만 아니라 Python 라이브러리로도 사용할 수 있습니다:

```python
from psyctl import DatasetBuilder, P2, LLMLoader, Settings
from pathlib import Path

# 설정 로드
settings = Settings()

# 모델 로더 생성
loader = LLMLoader()

# 데이터셋 빌더 생성
builder = DatasetBuilder()

# P2 클래스를 사용한 성격 프롬프트 생성
model, tokenizer = loader.load_model("google/gemma-3-270m-it")
p2 = P2(model, tokenizer)

# 성격별 캐릭터 설명 생성
extroverted_desc = p2.build("Alice", "Extroversion")
introverted_desc = p2.build("Alice", "Introversion")

print("외향적 Alice:", extroverted_desc)
print("내향적 Alice:", introverted_desc)

# CAA 데이터셋 생성
num_samples = builder.build_caa_dataset(
    model="google/gemma-3-270m-it",
    personality="Extroversion",
    output_dir=Path("./dataset"),
    limit_samples=100
)

print(f"생성된 샘플 수: {num_samples}")
```

#### 고급 사용 예시

```python
import psyctl
from psyctl import get_logger

# 로거 설정
logger = get_logger("my_app")

# 여러 성격 특성에 대한 데이터셋 생성
personalities = ["Extroversion", "Introversion", "Machiavellianism"]

for personality in personalities:
    logger.info(f"Creating dataset for {personality}")
    
    builder = psyctl.DatasetBuilder()
    num_samples = builder.build_caa_dataset(
        model="google/gemma-3-270m-it",
        personality=personality,
        output_dir=Path(f"./dataset/{personality.lower()}"),
        limit_samples=50
    )
    
    logger.success(f"Created {num_samples} samples for {personality}")
```

### 🤝 도움말

#### 도움말 보기

```bash
# 전체 도움말
psyctl --help

# 특정 명령어 도움말
psyctl dataset.build.caa --help
psyctl extract.steering --help
psyctl steering --help
psyctl benchmark --help
```

#### 버전 확인

```bash
psyctl --version
```

---

## 🔧 개발자 가이드

### 📁 프로젝트 구조

```
psyctl/
├── pyproject.toml              # 프로젝트 설정 및 의존성
├── README.md                   # 사용자 가이드
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
│       │   ├── prompt.py       # P2 구현
│       │   ├── utils.py
│       │   └── logger.py       # 로깅 설정
│       ├── models/             # 모델 관련
│       │   ├── llm_loader.py
│       │   └── vector_store.py
│       ├── data/               # 데이터 관련
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

### 🔄 개발 워크플로우

#### 1. 개발 환경 설정

```powershell
# 개발 환경 자동 설치
& .\scripts\install-dev.ps1
```

#### 2. 브랜치 생성

```bash
# 메인 브랜치에서 새 브랜치 생성
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

#### 3. 개발 및 테스트

```powershell
# 코드 포맷팅
& .\scripts\format.ps1

# 테스트 실행
& .\scripts\test.ps1

# 전체 빌드 프로세스 (포맷팅 + 린팅 + 테스트 + 설치)
& .\scripts\build.ps1
```

### 📜 개발 스크립트

프로젝트에는 개발 작업을 자동화하는 PowerShell 스크립트들이 포함되어 있습니다:

#### `install-dev.ps1` - 개발 환경 설치
```powershell
& .\scripts\install-dev.ps1
```
- uv 패키지 매니저 자동 설치
- 가상환경 생성 및 활성화
- 프로젝트 의존성 설치

#### `format.ps1` - 코드 포맷팅
```powershell
& .\scripts\format.ps1
```
- Black을 사용한 코드 포맷팅
- isort를 사용한 import 정렬
- `src/` 디렉토리 전체 적용

#### `test.ps1` - 테스트 실행
```powershell
& .\scripts\test.ps1
```
- pytest를 사용한 테스트 실행
- 커버리지 리포트 생성 (`htmlcov/` 디렉토리)
- 상세한 테스트 결과 출력

#### `build.ps1` - 전체 빌드 프로세스
```powershell
& .\scripts\build.ps1
```
- 코드 포맷팅 (Black + isort)
- 린팅 (flake8 + mypy)
- 테스트 실행 (pytest)
- 패키지 설치 (`uv pip install -e .`)

#### 3. 커밋 및 푸시

```bash
# 변경사항 스테이징
git add .

# 커밋
git commit -m "feat: add new feature description"

# 푸시
git push origin feature/your-feature-name
```

#### 4. Pull Request 생성

GitHub에서 Pull Request를 생성하고 다음을 포함하세요:
- 변경사항 설명
- 테스트 결과
- 관련 이슈 번호

### 📝 코딩 스타일

#### Python 코드 스타일

- **Black**: 코드 포맷팅
- **isort**: import 정렬
- **flake8**: 린팅
- **mypy**: 타입 체크

#### 명명 규칙

- **클래스**: PascalCase (`DatasetBuilder`)
- **함수/변수**: snake_case (`build_caa_dataset`)
- **상수**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`)
- **모듈**: snake_case (`dataset_builder.py`)

#### 문서화

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

### 🧪 테스트

#### 테스트 실행

```bash
# 모든 테스트 실행 (스크립트 사용 권장)
& .\scripts\test.ps1

# 또는 직접 실행
uv run pytest

# 특정 테스트 실행
uv run pytest tests/test_cli.py

# 커버리지와 함께 실행
uv run pytest --cov=psyctl --cov-report=html
```

#### 테스트 작성 가이드

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

### 🤝 기여 방법

#### 이슈 리포트

버그 리포트나 기능 요청 시 다음을 포함하세요:
- 문제/요청 설명
- 재현 단계
- 예상 동작
- 실제 동작
- 환경 정보 (OS, Python 버전 등)

#### 기능 개발

1. **이슈 생성**: 개발할 기능에 대한 이슈 생성
2. **브랜치 생성**: `feature/issue-number-description` 형식
3. **개발**: 기능 구현 및 테스트 작성
4. **테스트**: 모든 테스트 통과 확인
5. **문서화**: README나 API 문서 업데이트
6. **PR 생성**: Pull Request 생성

#### 버그 수정

1. **이슈 확인**: 기존 이슈가 있는지 확인
2. **브랜치 생성**: `fix/issue-number-description` 형식
3. **수정**: 버그 수정 및 테스트 추가
4. **검증**: 수정 사항이 다른 기능에 영향을 주지 않는지 확인
5. **PR 생성**: Pull Request 생성

### 📋 체크리스트

PR 제출 전 다음 사항을 확인하세요:

- [ ] 코드가 코딩 스타일을 준수하는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 새로운 기능에 대한 테스트가 작성되었는가?
- [ ] 문서가 업데이트되었는가?
- [ ] 커밋 메시지가 명확한가?
- [ ] PR 설명이 충분한가?

### 🚀 릴리스 프로세스

#### 버전 관리

- **Semantic Versioning** 사용 (MAJOR.MINOR.PATCH)
- `pyproject.toml`의 `version` 필드 업데이트
- 변경사항을 `CHANGELOG.md`에 기록

#### 릴리스 단계

1. **개발**: `main` 브랜치에서 개발
2. **테스트**: 모든 테스트 통과 확인
3. **버전 업데이트**: `pyproject.toml` 버전 수정
4. **태그 생성**: `git tag v1.0.0`
5. **배포**: GitHub Releases에 업로드

## Key papers
- [Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/pdf/2312.06681)
- [Steering Large Language Model Activations in Sparse Spaces](https://arxiv.org/pdf/2503.00177)
- [Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering](https://arxiv.org/pdf/2412.10427v1)
- [Toy model of superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Personalized Steering of LLMs: Versatile Steering Vectors via Bi-directional Preference Optimization](https://arxiv.org/abs/2406.00045)
- [The dark core of personality](https://psycnet.apa.org/record/2018-32574-001)
- [The Dark Triad of personality: Narcissism, Machiavellianism, and psychopathy. Journal of Research in Personality](https://www.sciencedirect.com/science/article/pii/S0092656602005056)
- [Style-Specific Neurons for Steering LLMs in Text Style Transfer](https://arxiv.org/abs/2410.00593)
- [Between facets and domains: 10 aspects of the Big Five. Journal of Personality and Social Psychology](https://psycnet.apa.org/fulltext/2007-15390-012.html)


## �� 라이센스

MIT License
