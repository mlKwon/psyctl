# PSYCTL - LLM Personality Steering Tool

> **⚠️ Project Under Development**  
> This project is currently under development and only supports limited functionality. Please check the release notes for stable features.

A project by [Persona Lab](https://modulabs.co.kr/labs/337) at ModuLabs.

A tool that supports steering LLMs to exhibit specific personalities. The goal is to automatically generate datasets and work with just a model and personality specification.



---

## 📚 Documentation

For detailed documentation on specific features:

- **[Build CAA Dataset](./docs/DATASET.BUILD.CAA.md)** - Complete guide to generating CAA datasets for steering vector extraction
- **[Extract Steering Vectors](./docs/EXTRACT.STEERING.md)** - Complete guide to extracting steering vectors using various methods

---

## 📖 User Guide

### 🚀 Quick Start

#### Installation

**Basic Installation (CPU Version)**
```bash
# Install uv (Windows)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# Project setup
uv venv
& .\.venv\Scripts\Activate.ps1
uv sync
```

**Installation in Google Colab**
```python
# Install directly from GitHub
!pip install git+https://github.com/modulabs-personalab/psyctl.git

# Or install from specific branch
!pip install git+https://github.com/modulabs-personalab/psyctl.git@main

# Set environment variables
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
os.environ['PSYCTL_LOG_LEVEL'] = 'INFO'

# Usage example
from psyctl import DatasetBuilder, P2, LLMLoader
```

**GPU Acceleration Installation (CUDA Support)**
```bash
# Install CUDA-enabled PyTorch after basic installation
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **Important**: The `transformers` package has `torch` as a dependency, so running `uv sync` will automatically install the CPU version. For GPU usage, you need to run the CUDA installation command above again.

#### Basic Usage

```bash
# 1. Generate dataset
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/cca"

# 2. Extract steering vector
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/cca" \
  --output "./steering_vector/out.safetensors"

# 3. Steering experiment
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello world blabla"

# 4. Inventory test
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory IPIP-NEO
```

### 📋 Detailed Command Guide

#### 1. Dataset Generation (`dataset.build.caa`)

Generates CAA datasets for steering vector extraction.

```bash
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/cca"
```

See [Build CAA Dataset](./docs/DATASET.BUILD.CAA.md) for detailed documentation.

#### 2. Steering Vector Extraction (`extract.steering`)

Extracts steering vectors from model activations.

```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/cca" \
  --output "./steering_vector/out.safetensors"
```

See [Extract Steering Vectors](./docs/EXTRACT.STEERING.md) for detailed documentation.

#### 3. Steering Experiment (`steering`)

Applies extracted steering vectors to generate text.

```bash
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello world blabla"
```

**Parameters:**
- `--model`: Model name to use
- `--steering-vector`: Steering vector file path
- `--input-text`: Input text

#### 4. Inventory Test (`benchmark`)

Measures personality changes using psychological inventories.

```bash
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory IPIP-NEO
```

**Parameters:**
- `--model`: Model name to use
- `--steering-vector`: Steering vector file path
- `--inventory`: Inventory name to use

### 📊 Supported Inventories

| Inventory | Domain | License | Notes |
|-----------|--------|---------|-------|
| IPIP-NEO-300/120 | Big Five | Public Domain | Full & short forms |
| NPI-40 | Narcissism | Free research use | Forced-choice |
| PNI-52 | Pathological narcissism | CC-BY-SA | Likert 1–6 |
| NARQ-18 | Admiration & Rivalry | CC-BY-NC | Two sub-scales |
| MACH-IV | Machiavellianism | Public Domain | Likert 1–5 |
| LSRP-26 | Psychopathy | Public Domain | Primary & secondary |
| PPI-56 | Psychopathy | Free research use | Short form |

### ⚙️ Configuration

PSYCTL uses environment variables for configuration. 

#### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | None | Hugging Face API token |
| `PSYCTL_OUTPUT_DIR` | `./output` | Output directory |
| `PSYCTL_DATASET_DIR` | `./dataset` | Dataset storage directory |
| `PSYCTL_STEERING_VECTOR_DIR` | `./steering_vector` | Steering vector storage |
| `PSYCTL_RESULTS_DIR` | `./results` | Results storage |
| `PSYCTL_CACHE_DIR` | `./temp` | Cache directory for models/datasets |
| `PSYCTL_LOG_LEVEL` | `INFO` | Logging level |
| `PSYCTL_LOG_FILE` | None | Log file path (optional) |
| **Batch Processing Settings** | | |
| `PSYCTL_INFERENCE_BATCH_SIZE` | `16` | Batch size for model inference |
| `PSYCTL_MAX_WORKERS` | `4` | Maximum number of worker threads |
| `PSYCTL_CHECKPOINT_INTERVAL` | `100` | Save checkpoint every N samples |

#### Setting Environment Variables

**Windows (PowerShell):**
```powershell
# Basic settings
$env:HF_TOKEN = "your_huggingface_token_here"
$env:PSYCTL_LOG_LEVEL = "DEBUG"

# Batch processing optimization
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"  # Increase for better GPU utilization
$env:PSYCTL_CHECKPOINT_INTERVAL = "50"   # Save checkpoints more frequently
```

**Linux/macOS:**
```bash
# Basic settings
export HF_TOKEN="your_huggingface_token_here"
export PSYCTL_LOG_LEVEL="DEBUG"

# Batch processing optimization
export PSYCTL_INFERENCE_BATCH_SIZE="32"  # Increase for better GPU utilization
export PSYCTL_CHECKPOINT_INTERVAL="50"   # Save checkpoints more frequently
```

#### Hugging Face Token Setup

Some models require a Hugging Face token for access:

1. Generate a token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Set environment variable:
   - `$env:HF_TOKEN="your_token_here"` (Windows)
   - `export HF_TOKEN="your_token_here"` (Linux/macOS)

#### Directory Configuration

All directories are automatically created when needed. You can customize paths using environment variables:

```powershell
# Custom directory configuration (Windows)
$env:PSYCTL_CACHE_DIR = "D:\ml_cache"
$env:PSYCTL_RESULTS_DIR = "C:\projects\results"
```

```bash
# Custom directory configuration (Linux/macOS)
export PSYCTL_CACHE_DIR="/data/ml_cache"
export PSYCTL_RESULTS_DIR="/projects/results"
```

#### Performance Optimization

**Batch Processing Optimization:**

The dataset generation now supports batch processing for significantly improved performance. Configure these settings based on your hardware:

```powershell
# For high-end GPUs (24GB+ VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"

# For mid-range GPUs (8-16GB VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "16"

# For low-end GPUs (4-8GB VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "8"

# Enable performance features
$env:PSYCTL_CHECKPOINT_INTERVAL = "100"  # Adjust based on stability needs
```

**Performance Tips:**
- Larger batch sizes improve GPU utilization but require more VRAM
- Checkpoint intervals of 50-100 samples balance performance and recovery
- Monitor GPU memory usage to find optimal batch size for your hardware

### 📝 Examples

#### Complete Workflow Example

```bash
# 1. Generate dataset for extroversion personality
# Set batch size for optimal performance
export PSYCTL_INFERENCE_BATCH_SIZE="16"

psyctl dataset.build.caa \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Extroversion" \
  --output "./dataset/extroversion" \
  --limit-samples 1000

# 2. Extract steering vector
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/extroversion" \
  --output "./steering_vector/extroversion.safetensors"

# 3. Apply steering to generate text
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --input-text "Tell me about yourself"

# 4. Measure personality changes
psyctl benchmark \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --inventory IPIP-NEO
```

#### Using as Python Library

PSYCTL can be used not only as a CLI tool but also as a Python library:

```python
from psyctl import DatasetBuilder, P2, LLMLoader, Settings
from pathlib import Path

# Load settings
settings = Settings()

# Create model loader
loader = LLMLoader()

# Create dataset builder
builder = DatasetBuilder()

# Generate personality prompts using P2 class
model, tokenizer = loader.load_model("google/gemma-3-270m-it")
p2 = P2(model, tokenizer)

# Generate character descriptions by personality
extroverted_desc = p2.build("Alice", "Extroversion")
introverted_desc = p2.build("Alice", "Introversion")

print("Extroverted Alice:", extroverted_desc)
print("Introverted Alice:", introverted_desc)

# Generate CAA dataset
num_samples = builder.build_caa_dataset(
    model="google/gemma-3-270m-it",
    personality="Extroversion",
    output_dir=Path("./dataset"),
    limit_samples=100
)

print(f"Generated samples: {num_samples}")
```

#### Advanced Usage Example

```python
import psyctl
from psyctl import get_logger

# Setup logger
logger = get_logger("my_app")

# Generate datasets for multiple personality traits
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

---

## 🔧 Developer Guide

### 📁 Project Structure

```
psyctl/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # User guide
├── .gitignore                  # Git ignore file
├── src/                        # Source code
│   └── psyctl/                 # Main package
│       ├── __init__.py
│       ├── cli.py              # CLI entry point
│       ├── config.py           # Configuration management
│       ├── commands/           # Command modules
│       │   ├── __init__.py
│       │   ├── dataset.py      # Dataset generation
│       │   ├── extract.py      # Steering vector extraction
│       │   ├── steering.py     # Steering experiments
│       │   └── benchmark.py    # Inventory tests
│       ├── core/               # Core logic
│       │   ├── __init__.py
│       │   ├── dataset_builder.py
│       │   ├── steering_extractor.py
│       │   ├── steering_applier.py
│       │   ├── inventory_tester.py
│       │   ├── prompt.py       # P2 implementation
│       │   ├── utils.py
│       │   └── logger.py       # Logging configuration
│       ├── models/             # Model-related
│       │   ├── __init__.py
│       │   ├── llm_loader.py
│       │   └── vector_store.py
│       └── data/               # Data-related
│           ├── __init__.py
│           └── inventories/    # Inventory data
│               ├── __init__.py
│               └── ipip_neo.py
├── tests/                      # Test code
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_commands/
│       ├── __init__.py
│       ├── test_dataset_builder.py
│       └── test_prompt.py
└── scripts/                    # Development scripts
    ├── install-dev.ps1
    ├── build.ps1
    ├── test.ps1
    └── format.ps1
```

### 🔄 Development Workflow

#### 1. Development Environment Setup

```powershell
# Automatic development environment installation
& .\scripts\install-dev.ps1
```

#### 2. Branch Creation

```bash
# Create new branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

#### 3. Development and Testing

```powershell
# Code formatting
& .\scripts\format.ps1

# Run tests
& .\scripts\test.ps1

# Complete build process (formatting + linting + testing + installation)
& .\scripts\build.ps1
```

### 📜 Development Scripts

The project includes PowerShell scripts to automate development tasks:

#### `install-dev.ps1` - Development Environment Installation
```powershell
& .\scripts\install-dev.ps1
```
- Automatic uv package manager installation
- Virtual environment creation and activation
- Project dependency installation

#### `format.ps1` - Code Formatting
```powershell
& .\scripts\format.ps1
```
- Code formatting using Black
- Import sorting using isort
- Applied to entire `src/` directory

#### `test.ps1` - Test Execution
```powershell
& .\scripts\test.ps1
```
- Test execution using pytest
- Coverage report generation (`htmlcov/` directory)
- Detailed test result output

#### `build.ps1` - Complete Build Process
```powershell
& .\scripts\build.ps1
```
- Code formatting (Black + isort)
- Linting (flake8 + mypy)
- Test execution (pytest)
- Package installation (`uv pip install -e .`)

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


