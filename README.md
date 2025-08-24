# PSYCTL - LLM Personality Steering Tool

A project by [Persona Lab](https://modulabs.co.kr/labs/337) at ModuLabs.

A tool that supports steering LLMs to exhibit specific personalities. The goal is to automatically generate datasets and work with just a model and personality specification.

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

Generates datasets for finding steering vectors.

```bash
psyctl dataset.build.caa \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/cca"
```

**Parameters:**
- `--model`: Model name to use (Hugging Face model ID)
- `--personality`: Target personality traits (comma-separated)
- `--output`: Dataset save path

#### 2. Steering Vector Extraction (`extract.steering`)

Extracts steering vectors using the CAA method.

```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/cca" \
  --output "./steering_vector/out.safetensors"
```

**Parameters:**
- `--model`: Model name to use
- `--layer`: Layer path to extract activations from
- `--dataset`: Dataset path
- `--output`: Steering vector save path (.safetensors)

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

#### Environment Variable Setup

You can set environment variables by creating a `.env` file in the project root:

```bash
# .env file example
PSYCTL_LOG_LEVEL=INFO
HF_TOKEN=your_huggingface_token_here
```

#### Log Level Configuration

You can set log levels through environment variables or `.env` file:

```bash
PSYCTL_LOG_LEVEL=DEBUG
```

#### Hugging Face Token Setup

Some models require a Hugging Face token for access:

1. Generate a token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Add `HF_TOKEN=your_token_here` to `.env` file
3. Or set as environment variable: `export HF_TOKEN=your_token_here`

#### Output Directories

The following directories are automatically created by default:
- `./dataset/` - Dataset storage
- `./steering_vector/` - Steering vector storage
- `./results/` - Results storage
- `./output/` - Other output files

### 📝 Examples

#### Complete Workflow Example

```bash
# 1. Generate dataset for extroversion personality
psyctl dataset.build.caa \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Extroversion" \
  --output "./dataset/extroversion"

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

### 🤝 Help

#### View Help

```bash
# General help
psyctl --help

# Specific command help
psyctl dataset.build.caa --help
psyctl extract.steering --help
psyctl steering --help
psyctl benchmark --help
```

#### Check Version

```bash
psyctl --version
```

#### Common Installation Issues

- **Dependency conflicts**: Run `pip install --upgrade pip` then reinstall
- **Permission issues**: Use `pip install --user`
- **Cache issues**: Run `pip cache purge` then reinstall

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
│       ├── commands/           # Command modules
│       │   ├── dataset.py      # Dataset generation
│       │   ├── extract.py      # Steering vector extraction
│       │   ├── steering.py     # Steering experiments
│       │   └── benchmark.py    # Inventory tests
│       ├── core/               # Core logic
│       │   ├── dataset_builder.py
│       │   ├── steering_extractor.py
│       │   ├── steering_applier.py
│       │   ├── inventory_tester.py
│       │   ├── prompt.py       # P2 implementation
│       │   ├── utils.py
│       │   └── logger.py       # Logging configuration
│       ├── models/             # Model-related
│       │   ├── llm_loader.py
│       │   └── vector_store.py
│       ├── data/               # Data-related
│       │   └── inventories/    # Inventory data
│       └── config/             # Configuration management
│           └── settings.py
├── tests/                      # Test code
│   ├── conftest.py
│   ├── test_cli.py
│   └── test_commands/
├── scripts/                    # Development scripts
│   ├── install-dev.ps1
│   ├── build.ps1
│   ├── test.ps1
│   └── format.ps1
└── docs/                       # Documentation
    └── README.md
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

#### 3. Commit and Push

```bash
# Stage changes
git add .

# Commit
git commit -m "feat: add new feature description"

# Push
git push origin feature/your-feature-name
```

#### 4. Pull Request Creation

Create a Pull Request on GitHub and include:
- Change description
- Test results
- Related issue number

### 📝 Coding Style

#### Python Code Style

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

#### Naming Conventions

- **Classes**: PascalCase (`DatasetBuilder`)
- **Functions/Variables**: snake_case (`build_caa_dataset`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MODEL`)
- **Modules**: snake_case (`dataset_builder.py`)

#### Documentation

- Write docstrings for all public functions and classes
- Use Google style docstrings
- Use type hints

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

### 🧪 Testing

#### Test Execution

```bash
# Run all tests (recommended to use script)
& .\scripts\test.ps1

# Or run directly
uv run pytest

# Run specific tests
uv run pytest tests/test_cli.py

# Run with coverage
uv run pytest --cov=psyctl --cov-report=html
```

#### Test Writing Guide

- Test filename: `test_*.py`
- Test function name: `test_*`
- Each test should be independent
- Use mocks to isolate external dependencies

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

### 🤝 Contribution Guidelines

#### Issue Reporting

When reporting bugs or requesting features, include:
- Problem/request description
- Reproduction steps
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)

#### Feature Development

1. **Create issue**: Create an issue for the feature to develop
2. **Create branch**: Use format `feature/issue-number-description`
3. **Development**: Implement feature and write tests
4. **Testing**: Ensure all tests pass
5. **Documentation**: Update README or API documentation
6. **Create PR**: Create Pull Request

#### Bug Fixes

1. **Check issues**: Check if issue already exists
2. **Create branch**: Use format `fix/issue-number-description`
3. **Fix**: Fix bug and add tests
4. **Verify**: Ensure fix doesn't affect other features
5. **Create PR**: Create Pull Request

### 📋 Checklist

Before submitting a PR, check the following:

- [ ] Does the code follow coding style?
- [ ] Do all tests pass?
- [ ] Are tests written for new features?
- [ ] Is documentation updated?
- [ ] Are commit messages clear?
- [ ] Is PR description sufficient?

### 🚀 Release Process

#### Version Management

- Use **Semantic Versioning** (MAJOR.MINOR.PATCH)
- Update `version` field in `pyproject.toml`
- Record changes in `CHANGELOG.md`

#### Release Steps

1. **Development**: Develop on `main` branch
2. **Testing**: Ensure all tests pass
3. **Version update**: Update version in `pyproject.toml`
4. **Create tag**: `git tag v1.0.0`
5. **Deploy**: Upload to GitHub Releases

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


## 📄 License

MIT License
