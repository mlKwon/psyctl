"""
Generate Korean Extroversion Dataset with Custom Templates
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

# Initialize logger
logger = get_logger("korean_extroversion")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Configuration
DATASET_MODEL = "moonshotai/kimi-k2-0905"
PERSONALITY = "외향성"  # Extroversion
SAMPLE_COUNT = 100
RESULTS_DIR = Path("./results")
DATASET_OUTPUT = RESULTS_DIR / "korean_extroversion_caa"

# Custom Korean templates for Extroversion
ROLEPLAY_TEMPLATE = """# 개요
이것은 롤플레이 세션입니다.
당신(어시스턴트 또는 모델)의 역할은 {{ char_name }}입니다.
사용자의 역할은 {{ user_name }}입니다.
{{ char_name }}의 짧은 반응을 한 문장으로 작성하세요.

# {{ char_name }}에 대하여
{{ p2 }}

# 상황
{{ situation }}
"""

CAA_QUESTION_TEMPLATE = """[상황]
{{ situation }}
[질문]
당신은 {{ char_name }}입니다. 이 상황에서 어떻게 반응하시겠습니까?
1. {{ answer_1 }}
2. {{ answer_2 }}
[답변]
"""

def main():
    """Generate Korean Extroversion dataset."""

    print("="*80)
    print("한국어 외향성(Extroversion) 데이터셋 생성")
    print("="*80)
    print(f"모델: {DATASET_MODEL}")
    print(f"성격: {PERSONALITY}")
    print(f"데이터셋: CaveduckAI/simplified_soda_kr (한국어 대화)")
    print(f"샘플 수: {SAMPLE_COUNT}")
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing DatasetBuilder with OpenRouter")
    dataset_builder = DatasetBuilder(
        use_openrouter=True,
        openrouter_api_key=OPENROUTER_API_KEY,
        openrouter_max_workers=4  # Parallel processing
    )

    # Set custom Korean templates
    logger.info("Setting custom Korean templates")
    dataset_builder.set_roleplay_prompt_template(ROLEPLAY_TEMPLATE)
    dataset_builder.set_caa_question_template(CAA_QUESTION_TEMPLATE)

    # Build CAA dataset
    logger.info("Starting CAA dataset generation")
    try:
        dataset_file = dataset_builder.build_caa_dataset(
            model=DATASET_MODEL,
            personality=PERSONALITY,
            output_dir=DATASET_OUTPUT,
            limit_samples=SAMPLE_COUNT,
            dataset_name="CaveduckAI/simplified_soda_kr",
            temperature=0.7,
            max_tokens=150
        )
        print(f"\n[성공] 데이터셋 생성 완료: {dataset_file}")
        logger.info(f"Dataset generated successfully: {dataset_file}")

        # Display sample data
        print("\n" + "-"*80)
        print("데이터셋 샘플 (처음 3개)")
        print("-"*80)
        import json
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                sample = json.loads(line)
                print(f"\n[샘플 {i+1}]")
                print(f"캐릭터: {sample['char_name']}")
                situation_preview = sample['situation'][:150]
                if len(sample['situation']) > 150:
                    situation_preview += "..."
                print(f"상황: {situation_preview}")
                print(f"긍정적(외향적): {sample['positive']}")
                print(f"중립적: {sample['neutral']}")
        print("-"*80)

        # Upload to HuggingFace Hub
        print("\n" + "="*80)
        print("HuggingFace Hub 업로드 시작")
        print("="*80)

        # Set metadata for upload
        dataset_builder.personality = PERSONALITY
        dataset_builder.active_model = DATASET_MODEL
        dataset_builder.dataset_name = "CaveduckAI/simplified_soda_kr"

        repo_url = dataset_builder.upload_to_hub(
            jsonl_file=dataset_file,
            repo_id="CaveduckAI/steer-personality-extroversion-ko",
            private=False,
            license="mit",
            token=HF_TOKEN
        )

        print(f"\n[성공] 데이터셋 업로드 완료!")
        print(f"저장소 URL: {repo_url}")
        print("="*80)

    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        raise

if __name__ == "__main__":
    main()
