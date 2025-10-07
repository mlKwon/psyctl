# Build CAA Dataset

This document describes how to build Contrastive Activation Addition (CAA) datasets using the `psyctl dataset.build.caa` command.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [OpenRouter Integration](#openrouter-integration)
- [Dataset Source](#dataset-source)
- [Output Format](#output-format)
- [Performance Optimization](#performance-optimization)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Adding Custom Datasets](#adding-custom-datasets)

## Overview

CAA datasets contain paired prompts designed to elicit contrasting personality-driven responses from language models. These datasets are used to extract steering vectors that can modify model behavior.

The dataset building process involves:

1. Loading a base conversational dataset (e.g., SODA)
2. Generating personality-specific character descriptions using P2 (Personality Prompt)
3. Creating positive/neutral prompt pairs for each scenario
4. Saving the dataset in JSONL format for steering vector extraction

## Usage

### Basic Command

Generate a CAA dataset for a specific personality trait:

```bash
psyctl dataset.build.caa \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/extroversion"
```

### With Custom Dataset

Use a different Hugging Face dataset as the base:

```bash
psyctl dataset.build.caa \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Machiavellianism" \
  --dataset "username/custom-conversations" \
  --output "./dataset/machiavellianism"
```

### Limit Sample Count

Generate a specific number of samples for testing:

```bash
psyctl dataset.build.caa \
  --model "google/gemma-3-270m-it" \
  --personality "Extroversion" \
  --output "./dataset/test" \
  --limit-samples 100
```

### Multiple Personalities

Generate datasets for multiple personality traits:

```bash
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellianism" \
  --output "./dataset/multi"
```

## OpenRouter Integration

PSYCTL supports OpenRouter API for dataset generation without local GPU requirements. See [OpenRouter Guide](./OPENROUTER.md) for detailed documentation.

### CLI Usage

#### Basic OpenRouter Usage

Generate dataset using OpenRouter API:

```bash
psyctl dataset.build.caa \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --personality "Extroversion" \
  --output "./dataset/openrouter"
```

#### With Custom Model

```bash
psyctl dataset.build.caa \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-model "meta-llama/llama-3.1-405b-instruct" \
  --personality "Machiavellianism" \
  --output "./dataset/llama"
```

#### Parallel Processing

Speed up generation with multiple workers:

```bash
psyctl dataset.build.caa \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-max-workers 4 \
  --personality "Extroversion" \
  --output "./dataset/fast" \
  --limit-samples 1000
```

### Programmatic Usage (Python API)

Use DatasetBuilder directly in Python code with OpenRouter:

#### Basic Usage

```python
from psyctl import DatasetBuilder
from pathlib import Path

# Initialize DatasetBuilder with OpenRouter
dataset_builder = DatasetBuilder(
    use_openrouter=True,
    openrouter_api_key="sk-or-v1-xxxx"
)

# Build CAA dataset
num_samples = dataset_builder.build_caa_dataset(
    model="",  # Not used in OpenRouter mode, can be empty
    personality="Extroversion",
    output_dir=Path("./dataset/openrouter"),
    limit_samples=1000
)

print(f"Generated {num_samples} samples")
```

#### With Custom Model and Dataset

```python
from psyctl import DatasetBuilder
from pathlib import Path

# Initialize with custom OpenRouter model
dataset_builder = DatasetBuilder(
    use_openrouter=True,
    openrouter_api_key="sk-or-v1-xxxx",
    openrouter_model="meta-llama/llama-3.1-405b-instruct"
)

# Build CAA dataset with custom Hugging Face dataset
num_samples = dataset_builder.build_caa_dataset(
    model="",  # Not used in OpenRouter mode
    personality="Machiavellianism",
    output_dir=Path("./dataset/custom"),
    limit_samples=500,
    dataset_name="CaveduckAI/simplified_soda_kr"
)

print(f"Generated {num_samples} samples")
```

#### With Parallel Processing

```python
from psyctl import DatasetBuilder
from pathlib import Path

# Initialize with parallel workers
dataset_builder = DatasetBuilder(
    use_openrouter=True,
    openrouter_api_key="sk-or-v1-xxxx",
    openrouter_model="google/gemma-2-27b-it",
    openrouter_max_workers=4  # Process 4 requests in parallel
)

# Build dataset
num_samples = dataset_builder.build_caa_dataset(
    model="",
    personality="Extroversion",
    output_dir=Path("./dataset/parallel"),
    limit_samples=1000
)

print(f"Generated {num_samples} samples using parallel processing")
```

#### Environment Variable for API Key

```python
import os
from psyctl import DatasetBuilder
from pathlib import Path

# Set API key via environment variable
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-xxxx"

# Initialize without explicit API key
dataset_builder = DatasetBuilder(
    use_openrouter=True,
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
)

# Build dataset
num_samples = dataset_builder.build_caa_dataset(
    model="",
    personality="Extroversion",
    output_dir=Path("./dataset/openrouter"),
    limit_samples=1000
)
```

#### Constructor Parameters

**DatasetBuilder() Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_openrouter` | bool | False | Enable OpenRouter mode |
| `openrouter_api_key` | str | None | OpenRouter API key (required if use_openrouter=True) |
| `openrouter_model` | str | "qwen/qwen3-next-80b-a3b-instruct" | OpenRouter model identifier |
| `openrouter_max_workers` | int | 1 | Number of parallel workers for API calls |

**build_caa_dataset() Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | str | Yes | Model identifier (not used in OpenRouter mode, pass "") |
| `personality` | str | Yes | Target personality trait |
| `output_dir` | Path | Yes | Output directory for dataset |
| `limit_samples` | int | Yes | Maximum number of samples to generate |
| `dataset_name` | str | No | Hugging Face dataset name (default: "allenai/soda") |

### OpenRouter vs Local Model

| Feature | OpenRouter | Local Model |
|---------|-----------|-------------|
| GPU Required | No | Yes |
| Cost | Per API call | Free (after hardware) |
| Model Size | Up to 405B+ | Limited by VRAM |
| Speed (small datasets) | Slower | Faster |
| Speed (with parallel) | Competitive | Faster |
| Setup Time | Instant | Model download required |

**When to use OpenRouter:**
- No GPU available
- Need large models (70B+, 405B)
- Testing different models
- One-time dataset generation

**When to use Local Model:**
- GPU available
- Frequent dataset generation
- Large-scale production
- Cost optimization

### Command-Line Options

**Required Options:**
- `--personality`: Target personality traits, comma-separated (required)
- `--output`: Output directory path (required)

**Model Options:**
- `--model`: Hugging Face model identifier (required for local mode)
- `--openrouter-api-key`: OpenRouter API key for cloud mode (alternative to --model)
- `--openrouter-model`: OpenRouter model identifier (default: qwen/qwen3-next-80b-a3b-instruct)
- `--openrouter-max-workers`: Number of parallel workers for OpenRouter (default: 1)

**Dataset Options:**
- `--dataset`: Hugging Face dataset name (default: "allenai/soda")
- `--limit-samples`: Maximum number of samples to generate (default: all)

**Template Options:**
- `--caa-question-template`: Path to custom Jinja2 template for CAA questions (.j2 file)
- `--roleplay-prompt-template`: Path to custom Jinja2 template for roleplay prompts (.j2 file)

**Performance Options:**
- `--batch-size`: Batch size for inference (default: from config, local mode only)

## Dataset Source

### Default Dataset: SODA

By default, the command uses the [SODA dataset](https://huggingface.co/datasets/allenai/soda) (Social Dialogue dataset) which contains:

- Over 1.5M dialogue turns
- Diverse conversational scenarios
- Natural social interaction patterns
- High-quality human annotations

### Using Custom Datasets

You can use any Hugging Face dataset that provides conversational contexts:

**Requirements:**
- Must be accessible via Hugging Face Datasets library
- Should contain dialogue or scenario information
- Recommended: Conversational or social interaction data

**Example custom datasets:**
```bash
# Using DailyDialog
psyctl dataset.build.caa \
  --dataset "daily_dialog" \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/daily"

# Using custom dataset
psyctl dataset.build.caa \
  --dataset "myusername/my-conversations" \
  --model "google/gemma-2-2b-it" \
  --personality "Introversion" \
  --output "./dataset/custom"
```

## Output Format

### Directory Structure

```
output_directory/
├── caa_dataset_20250107_143022.jsonl            # Main dataset file (timestamped)
└── caa_dataset_20250107_143022.checkpoint.json  # Checkpoint file
```

### JSONL Format

Each line in the dataset file contains:

```json
{
  "question": "[Situation]\nAlice is at a party...\n[Question]\nWhat should Alice say?\n1. Let's all dance together!\n2. I prefer to observe quietly.\n[Answer]",
  "positive": "(1",
  "neutral": "(2"
}
```

**Fields:**
- `question`: The scenario description with answer options
- `positive`: Answer option exhibiting the target personality
- `neutral`: Answer option with neutral personality expression

### Checkpoint Format

The checkpoint file contains:

```json
{
  "num_generated": 500,
  "output_file": "c:/work/psyctl/dataset/caa_dataset_20250107_143022.jsonl",
  "timestamp": "2025-01-07T14:35:22.123456"
}
```

## Performance Optimization

### Batch Processing

The dataset builder uses batch processing for improved GPU utilization:

**Configure batch size:**
```powershell
# Windows
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"

# Linux/macOS
export PSYCTL_INFERENCE_BATCH_SIZE="32"
```

**Recommended batch sizes:**
- High-end GPUs (24GB+ VRAM): 32-64
- Mid-range GPUs (8-16GB VRAM): 16-32
- Low-end GPUs (4-8GB VRAM): 8-16
- CPU: 4-8


## Checkpoint and Resume

### Automatic Checkpointing

The dataset builder automatically saves checkpoints during generation:

**Default behavior:**
- Checkpoint saved every 100 samples (configurable)
- Checkpoint file: `caa_dataset_{timestamp}.checkpoint.json`
- Output file: `caa_dataset_{timestamp}.jsonl`

**Configure checkpoint interval:**
```powershell
# Save checkpoint every 50 samples
$env:PSYCTL_CHECKPOINT_INTERVAL = "50"

# Save checkpoint every 200 samples
$env:PSYCTL_CHECKPOINT_INTERVAL = "200"
```

### Resume from Checkpoint

If dataset generation is interrupted, simply re-run the same command:

```bash
# Original command
psyctl dataset.build.caa \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/extroversion"

# After interruption, run the same command
# It will automatically detect and resume from the latest checkpoint
```

**Resume behavior:**
- Automatically detects existing checkpoints
- Loads progress from latest checkpoint
- Continues from last saved sample
- Preserves all previous work

### Manual Checkpoint Management

**Check checkpoint status:**
```powershell
# Windows
dir ./dataset/extroversion/*.checkpoint.json

# Linux/macOS
ls ./dataset/extroversion/*.checkpoint.json
```

**View checkpoint contents:**
```powershell
# Check number of samples generated
type ./dataset/extroversion/caa_dataset_*.checkpoint.json
```

## Adding Custom Datasets

To use a custom Hugging Face dataset as the source:

### 1. Upload Dataset to Hugging Face

```python
from datasets import Dataset
import pandas as pd

# Create your conversational dataset
data = {
    'dialogue': ['conversation 1...', 'conversation 2...'],
    'narrative': ['narrative 1...', 'narrative 2...'],
    # ... other fields
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Upload to Hugging Face
dataset.push_to_hub("username/my-conversations")
```

### 2. Use in Dataset Building

```bash
psyctl dataset.build.caa \
  --dataset "username/my-conversations" \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/custom"
```

### 3. Dataset Requirements

**Minimum requirements:**
- Accessible via Hugging Face Datasets
- Contains conversational or scenario data
- Sufficient samples for meaningful extraction

**Recommended characteristics:**
- Diverse scenarios covering various social situations
- Natural language patterns
- 1000+ samples for robust steering vectors
- Clean, well-formatted data

### 4. Testing with Small Samples

Test your custom dataset with limited samples first:

```bash
psyctl dataset.build.caa \
  --dataset "username/my-conversations" \
  --model "google/gemma-3-270m-it" \
  --personality "Extroversion" \
  --output "./dataset/test" \
  --limit-samples 10
```

## Implementation Details

### P2 (Personality Prompt) Generation

The dataset builder uses the P2 class to generate personality-specific character descriptions:

**Algorithm:**
1. Extract character name from scenario
2. Generate positive personality description
3. Generate neutral personality description
4. Create answer options based on descriptions
5. Format as CAA prompt pair

**Example:**
```
Input: "Alice is at a party..."
Personality: "Extroversion"

Positive Description:
"Alice is outgoing, energetic, and loves socializing..."

Neutral Description:
"Alice is balanced, comfortable in various settings..."

Output Options:
1. [Extroverted response]
2. [Neutral response]
```

### Batch Processing Pipeline

1. **Load base dataset** (SODA or custom)
2. **Batch scenarios** into groups
3. **Generate P2 descriptions** in parallel
4. **Format CAA pairs** with answer options
5. **Save to JSONL** with checkpoints
6. **Update metadata** with generation stats

## Troubleshooting

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size:
```powershell
$env:PSYCTL_INFERENCE_BATCH_SIZE = "8"
```

### Dataset Not Found

```
Error: Dataset 'username/dataset-name' not found
```

**Solution:** Verify dataset exists and is public:
1. Check dataset URL: `https://huggingface.co/datasets/username/dataset-name`
2. Ensure dataset is public or HF_TOKEN is set
3. Verify dataset name spelling

### Checkpoint Corruption

```
Error: Failed to load checkpoint
```

**Solution:** Remove corrupted checkpoint and restart:
```powershell
# Windows
del ./dataset/extroversion/*.checkpoint.json
psyctl dataset.build.caa --model ... --personality ... --output ...

# Linux/macOS
rm ./dataset/extroversion/*.checkpoint.json
psyctl dataset.build.caa --model ... --personality ... --output ...
```

### Slow Generation Speed

**Optimization checklist:**
1. Increase batch size (if GPU memory allows)
2. Use faster/smaller model for testing
3. Check GPU utilization
4. Enable checkpointing for long runs

```powershell
# Performance tuning
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"
$env:PSYCTL_CHECKPOINT_INTERVAL = "100"
```

## Best Practices

### Model Selection

**For dataset generation:**
- Use instruction-tuned models (e.g., `*-it`, `*-Instruct`)
- Larger models (7B+) produce higher quality descriptions
- Test with small models first (270M-2B)

**Recommended models:**
- `google/gemma-2-2b-it` (fast, good quality)
- `google/gemma-3-27b-it` (high quality)
- `meta-llama/Llama-3.2-3B-Instruct` (balanced)

### Personality Traits

**Single trait (recommended):**
```bash
--personality "Extroversion"
```

**Multiple traits:**
```bash
--personality "Extroversion, Machiavellianism"
```

### Custom Templates

**Using Jinja2 Templates:**

PSYCTL supports custom Jinja2 templates to customize prompt generation. This allows you to:
- Support different languages
- Customize prompt styles
- Adapt to specific use cases

**Template Types:**

1. **CAA Question Template** (`caa_question.j2`): Formats the final question with situation and answer options
2. **Roleplay Prompt Template** (`roleplay_prompt.j2`): Generates character responses

**Default Templates Location:**
```
src/psyctl/templates/
├── caa_question.j2
└── roleplay_prompt.j2
```

**Creating Custom Templates:**

1. Create a custom CAA question template:

```jinja2
# custom_caa.j2
[상황]
{{ situation }}
[질문]
당신은 {{ char_name }}입니다. 이 상황에서 어떻게 반응하시겠습니까?
1. {{ answer_1 }}
2. {{ answer_2 }}
[답변]
```

2. Create a custom roleplay prompt template:

```jinja2
# custom_roleplay.j2
# 개요
이것은 롤플레이 세션입니다.
당신(어시스턴트 또는 모델)의 역할은 {{ char_name }}입니다.
사용자의 역할은 {{ user_name }}입니다.
{{ char_name }}의 짧은 반응을 한 문장으로 작성하세요.

# {{ char_name }}에 대하여
{{ p2 }}

# 상황
{{ situation }}
```

3. Use custom templates:

```bash
psyctl dataset.build.caa \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/korean" \
  --caa-question-template "./templates/custom_caa.j2" \
  --roleplay-prompt-template "./templates/custom_roleplay.j2"
```

**Available Template Variables:**

**CAA Question Template:**
- `{{ char_name }}`: Character name
- `{{ situation }}`: Conversation context
- `{{ answer_1 }}`: First answer option
- `{{ answer_2 }}`: Second answer option

**Roleplay Prompt Template:**
- `{{ user_name }}`: User/asker name
- `{{ char_name }}`: Character name
- `{{ p2 }}`: Personality description
- `{{ situation }}`: Conversation context

**Programmatic Usage:**

```python
from psyctl import DatasetBuilder
from pathlib import Path

# Initialize with custom templates
builder = DatasetBuilder(
    caa_question_template="./templates/custom_caa.j2",
    roleplay_prompt_template="./templates/custom_roleplay.j2"
)

# Build dataset
output_file = builder.build_caa_dataset(
    model="google/gemma-2-2b-it",
    personality="Extroversion",
    output_dir=Path("./dataset/custom"),
    limit_samples=100
)
```
**Dynamic Template Management:**

You can also get and set templates dynamically at runtime without file operations:

```python
from psyctl import DatasetBuilder

builder = DatasetBuilder()

# Get current template as string
current_template = builder.get_caa_question_template()
print(current_template)

# Modify and set new template
modified_template = current_template.replace("[Situation]", "[Scenario]")
builder.set_caa_question_template(modified_template)

# Verify the change
new_template = builder.get_caa_question_template()
assert "[Scenario]" in new_template

# Now build dataset with modified template
output_file = builder.build_caa_dataset(
    model="google/gemma-2-2b-it",
    personality="Extroversion",
    output_dir=Path("./dataset/modified"),
    limit_samples=100
)
```

**Template Management Methods:**

| Method | Description |
|--------|-------------|
| `get_caa_question_template()` | Get current CAA question template as string |
| `get_roleplay_prompt_template()` | Get current roleplay prompt template as string |
| `set_caa_question_template(template_str)` | Set CAA question template from string |
| `set_roleplay_prompt_template(template_str)` | Set roleplay prompt template from string |


## References

- [SODA Dataset](https://huggingface.co/datasets/allenai/soda)
- [CAA Paper: Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [P2 Paper: Evaluating and Inducing Personality](https://arxiv.org/abs/2206.07550)
- [PSYCTL Steering Extraction](./EXTRACT.STEERING.md)
