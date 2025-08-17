# PSYCTL - LLM Personality Steering Tool

LLM을 지정된 성격으로 steering 하는 것을 지원하는 툴입니다. 자동으로 데이터셋을 생성하여 모델과 성격만 지정하면 작동하는 것이 목표입니다.

## 🚀 빠른 시작

### 설치

```bash
# uv 설치 (Windows)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# 프로젝트 설정
uv venv
& .\.venv\Scripts\Activate.ps1
uv sync
```

### 기본 사용법

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

## 📖 상세 사용법

### 1. 데이터셋 생성 (`dataset.build.caa`)

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

### 2. 스티어링 벡터 추출 (`extract.steering`)

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

### 3. 스티어링 실험 (`steering`)

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

### 4. 인벤토리 테스트 (`benchmark`)

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

## 📊 지원하는 인벤토리

| Inventory | Domain | License | Notes |
|-----------|--------|---------|-------|
| IPIP-NEO-300/120 | Big Five | Public Domain | Full & short forms |
| NPI-40 | Narcissism | Free research use | Forced-choice |
| PNI-52 | Pathological narcissism | CC-BY-SA | Likert 1–6 |
| NARQ-18 | Admiration & Rivalry | CC-BY-NC | Two sub-scales |
| MACH-IV | Machiavellianism | Public Domain | Likert 1–5 |
| LSRP-26 | Psychopathy | Public Domain | Primary & secondary |
| PPI-56 | Psychopathy | Free research use | Short form |

## 🔧 설정

### 로그 레벨 설정

환경 변수나 `.env` 파일을 통해 로그 레벨을 설정할 수 있습니다:

```bash
PSYCTL_LOG_LEVEL=DEBUG
```

### 출력 디렉토리

기본적으로 다음 디렉토리들이 자동으로 생성됩니다:
- `./dataset/` - 데이터셋 저장
- `./steering_vector/` - 스티어링 벡터 저장
- `./results/` - 결과 저장
- `./output/` - 기타 출력 파일

## 📝 예시

### 전체 워크플로우 예시

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

## 🤝 도움말

### 도움말 보기

```bash
# 전체 도움말
psyctl --help

# 특정 명령어 도움말
psyctl dataset.build.caa --help
psyctl extract.steering --help
psyctl steering --help
psyctl benchmark --help
```

### 버전 확인

```bash
psyctl --version
```

## 📚 추가 문서

- [개발자 가이드](CONTRIBUTING.md) - 개발 환경 설정 및 기여 방법
- [API 문서](docs/README.md) - 상세한 API 문서
- [예시 및 튜토리얼](docs/examples/) - 다양한 사용 예시

## 📄 라이센스

MIT License
