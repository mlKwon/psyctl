# 변경사항 기록

이 파일은 PSYCTL 프로젝트의 모든 중요한 변경사항을 기록합니다.

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따르며,
이 프로젝트는 [Semantic Versioning](https://semver.org/lang/ko/)을 준수합니다.

## [Unreleased]

### 추가됨
- 프로젝트 초기 구조 설정
- CLI 명령어 기본 구조
- 로깅 시스템 (loguru) 통합
- PowerShell 스크립트 (개발 환경 설정, 빌드, 테스트, 포맷팅)
- 기본 테스트 구조
- 문서화 (README.md, CONTRIBUTING.md)

### 변경됨
- 없음

### 제거됨
- 없음

### 수정됨
- 없음

### 보안
- 없음

## [0.1.0] - 2024-01-15

### 추가됨
- 초기 프로젝트 릴리스
- 기본 CLI 구조
- 4가지 주요 명령어:
  - `dataset.build.caa`: CAA 데이터셋 생성
  - `extract.steering`: 스티어링 벡터 추출
  - `steering`: 스티어링 실험
  - `benchmark`: 인벤토리 테스트
- 로깅 시스템
- PowerShell 개발 스크립트
- 기본 테스트 구조
- 문서화

### 의존성
- click>=8.0.0: CLI 프레임워크
- rich>=13.0.0: 터미널 출력 개선
- pydantic>=2.0.0: 데이터 검증
- torch>=2.0.0: PyTorch
- transformers>=4.30.0: Hugging Face Transformers
- safetensors>=0.3.0: SafeTensors
- datasets>=2.10.0: Hugging Face Datasets
- numpy>=1.24.0: NumPy
- pandas>=2.0.0: Pandas
- scikit-learn>=1.3.0: Scikit-learn
- matplotlib>=3.7.0: Matplotlib
- seaborn>=0.12.0: Seaborn
- loguru>=0.7.0: 로깅

### 개발 의존성
- pytest>=7.0.0: 테스트 프레임워크
- pytest-cov>=4.0.0: 커버리지
- pytest-mock>=3.10.0: Mock
- black>=23.0.0: 코드 포맷팅
- isort>=5.0.0: import 정렬
- flake8>=6.0.0: 린팅
- mypy>=1.0.0: 타입 체크
- jupyter>=1.0.0: Jupyter

---

## 버전 형식

- **MAJOR**: 호환되지 않는 API 변경
- **MINOR**: 이전 버전과 호환되는 기능 추가
- **PATCH**: 이전 버전과 호환되는 버그 수정

## 변경 유형

- **추가됨**: 새로운 기능
- **변경됨**: 기존 기능의 변경
- **제거됨**: 기능 제거
- **수정됨**: 버그 수정
- **보안**: 보안 관련 변경사항

## 기여하기

변경사항을 기록할 때는 다음을 고려해주세요:

1. **명확성**: 변경사항을 명확하고 간결하게 설명
2. **사용자 중심**: 사용자 관점에서 중요한 변경사항 강조
3. **구체성**: 가능한 한 구체적인 예시나 링크 포함
4. **일관성**: 기존 형식과 일관성 유지
