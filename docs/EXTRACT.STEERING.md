# Extract Steering Vectors

This document describes how to extract steering vectors from language models using the `psyctl extract.steering` command.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Extraction Methods](#extraction-methods)
- [Multi-Layer Extraction](#multi-layer-extraction)
- [Output Format](#output-format)
- [Adding New Extraction Methods](#adding-new-extraction-methods)

## Overview

Steering vectors are learned representations that can modify language model behavior to exhibit specific personality traits or characteristics. The extraction process involves:

1. Loading a CAA (Contrastive Activation Addition) dataset with positive/neutral prompt pairs
2. Running inference on the model to collect internal activations
3. Computing steering vectors from the activation differences
4. Saving vectors in safetensors format for later use

## Usage

### CLI Usage

#### Basic Single-Layer Extraction

Extract a steering vector from a single model layer:

```bash
psyctl extract.steering \
  --model "google/gemma-2-2b-it" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors"
```

#### Multi-Layer Extraction

Extract steering vectors from multiple layers simultaneously:

```bash
# Using repeated --layer flags
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --layer "model.layers[14].mlp.down_proj" \
  --layer "model.layers[15].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/multi_layer.safetensors"

# Or using comma-separated values
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj,model.layers[15].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/multi_layer.safetensors"
```

#### Command-Line Options

- `--model`: Hugging Face model identifier (required)
- `--layer`: Single layer path (can be repeated for multi-layer extraction)
- `--layers`: Comma-separated list of layer paths
- `--dataset`: Path to CAA dataset directory containing JSONL file (required)
- `--output`: Output path for safetensors file (required)
- `--batch-size`: Batch size for inference (default: from config)
- `--normalize`: Normalize steering vectors to unit length (optional)

### Python Code Usage

You can use the `SteeringExtractor` class directly in Python code:

#### Basic Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

# Initialize extractor
extractor = SteeringExtractor()

# Extract steering vector using CAA method
vectors = extractor.extract_caa(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp.down_proj"],
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/bipo_steering.safetensors"),
    normalize=False,
    method="mean_contrastive"
)

# vectors is a dict: {"model.layers.13.mlp.down_proj": torch.Tensor}
print(f"Extracted {len(vectors)} vectors")
for layer_name, vector in vectors.items():
    print(f"  {layer_name}: shape={vector.shape}, norm={vector.norm().item():.4f}")
```

#### BiPO Method Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

extractor = SteeringExtractor()

# Extract using BiPO optimization method
vectors = extractor.extract_caa(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp"],  # BiPO uses layer modules, not projections
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/bipo_steering.safetensors"),
    batch_size=16,
    normalize=False,
    method="bipo",
    lr=5e-4,
    beta=0.1,
    epochs=10
)
```

#### Multi-Layer Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

extractor = SteeringExtractor()

# Extract from multiple layers simultaneously
layers = [
    "model.layers.10.mlp.down_proj",
    "model.layers.12.mlp.down_proj",
    "model.layers.14.mlp.down_proj",
]

vectors = extractor.extract_caa(
    model_name="google/gemma-2-2b-it",
    layers=layers,
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/multi_layer_steering.safetensors"),
    batch_size=16,
    normalize=True,
    method="mean_contrastive"
)

# Analyze extracted vectors
for layer_name, vector in vectors.items():
    norm = vector.norm().item()
    mean = vector.mean().item()
    std = vector.std().item()
    print(f"{layer_name}:")
    print(f"  Shape: {vector.shape}")
    print(f"  Norm: {norm:.4f}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
```

#### Loading and Using Extracted Vectors

```python
from safetensors.torch import load_file
import json

# Load saved vectors
data = load_file("./results/bipo_steering.safetensors")

# Access vectors
for key, tensor in data.items():
    if key != "__metadata__":
        print(f"Layer: {key}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")

# Load metadata
metadata_json = data.get("__metadata__", {})
if isinstance(metadata_json, bytes):
    metadata = json.loads(metadata_json.decode('utf-8'))
elif isinstance(metadata_json, str):
    metadata = json.loads(metadata_json)
else:
    metadata = metadata_json

print(f"\nMetadata:")
print(f"  Model: {metadata.get('model')}")
print(f"  Method: {metadata.get('method')}")
print(f"  Layers: {metadata.get('num_layers')}")
print(f"  Normalized: {metadata.get('normalized')}")
print(f"  Dataset samples: {metadata.get('dataset_samples')}")
```

#### Complete Workflow Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor
from safetensors.torch import load_file

# 1. Extract steering vector
extractor = SteeringExtractor()

vectors = extractor.extract_caa(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp.down_proj"],
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/extroversion_steering.safetensors"),
    batch_size=16,
    normalize=False,
    method="mean_contrastive"
)

# 2. Verify extraction
layer_name = "model.layers.13.mlp.down_proj"
vector = vectors[layer_name]
print(f"Extracted vector: shape={vector.shape}, norm={vector.norm().item():.4f}")

# 3. Load for later use
loaded_data = load_file("./results/extroversion_steering.safetensors")
loaded_vector = loaded_data[layer_name]
print(f"Loaded vector: shape={loaded_vector.shape}")

# 4. Apply to model (see STEERING.md for details)
# ...
```

### Layer Path Format

Layer paths use dot notation with bracket indexing:

```
model.layers[13].mlp.down_proj
model.layers[0].self_attn.o_proj
model.language_model.layers[10].mlp.act_fn
```

Common layer targets:
- `mlp.down_proj`: MLP output projection (recommended)
- `mlp.act_fn`: After activation function
- `self_attn.o_proj`: Attention output projection

## Extraction Methods

### MeanContrastiveActivationVector (CAA)

The CAA extraction method computes steering vectors as the mean difference between positive and neutral activations:

**Algorithm:**
1. Load CAA dataset containing positive/neutral prompt pairs
2. For each layer:
   - Collect activations from positive prompts
   - Collect activations from neutral prompts
   - Compute incremental means (memory efficient)
3. Calculate: `steering_vector = mean(positive_activations) - mean(neutral_activations)`
4. Optionally normalize to unit length

**Example:**
```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors" \
  --method mean_contrastive
```

### BiPO (Bi-Directional Preference Optimization)

BiPO is an optimization-based method that learns steering vectors through preference learning:

**Algorithm:**
1. Load CAA dataset containing positive/neutral prompt pairs
2. Initialize learnable steering parameters for each layer
3. For each training epoch:
   - Apply steering to model activations
   - Compute preference loss between positive/neutral outputs
   - Update steering parameters via gradient descent
4. Extract final optimized steering vectors
5. Optionally normalize to unit length

**Hyperparameters:**
- `--lr`: Learning rate (default: 5e-4)
- `--beta`: Beta parameter for preference loss (default: 0.1)
- `--epochs`: Number of training epochs (default: 10)

**Example:**
```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors" \
  --method bipo \
  --lr 5e-4 \
  --beta 0.1 \
  --epochs 10
```

**Note:** BiPO requires layer modules (e.g., `model.layers[13].mlp`) rather than specific projections (e.g., `model.layers[13].mlp.down_proj`).

### Method Comparison

| Feature | CAA (mean_contrastive) | BiPO |
|---------|------------------------|------|
| Speed | Fast | Slower (optimization) |
| Resource Usage | Low | Higher (training) |
| Hyperparameters | None | lr, beta, epochs |
| Use Case | Quick steering | High-quality steering |

## Multi-Layer Extraction

Extracting from multiple layers simultaneously offers several advantages:

**Benefits:**
1. **Efficiency**: Single forward pass collects activations from all layers
2. **Consistency**: All vectors extracted from same dataset samples
3. **Experimentation**: Compare steering strength across layers
4. **Ensemble**: Combine vectors from multiple layers during application

**Best Practices:**
- Test layers in middle-to-late transformer blocks (e.g., layers 10-20 for 24-layer models)
- Focus on MLP output projections (`mlp.down_proj`)
- Extract 3-5 consecutive layers for comparison
- Use visualization tools to analyze vector magnitudes across layers

## Output Format

Steering vectors are saved in safetensors format with embedded metadata:

```python
# File structure
{
    "model.layers[13].mlp.down_proj": torch.Tensor,  # First layer's steering vector
    "model.layers[14].mlp.down_proj": torch.Tensor,  # Second layer's steering vector
    # ... more layers
    "__metadata__": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "method": "mean_contrastive",  # or "bipo"
        "layers": ["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"],
        "dataset_path": "./dataset/caa",
        "dataset_samples": 20000,
        "num_layers": 2,
        "normalized": false
    }
}
```

**Loading vectors:**

```python
from safetensors.torch import load_file

data = load_file("steering_vector.safetensors")
layer_13_vector = data["model.layers[13].mlp.down_proj"]
metadata = data["__metadata__"]
```

## Adding New Extraction Methods

To implement a new steering vector extraction method, follow these steps:

### 1. Create Extractor Class

Create a new file in `src/psyctl/core/extractors/`:

```python
# src/psyctl/core/extractors/my_method_extractor.py

from typing import Dict
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer

from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.logger import get_logger


class MyMethodExtractor(BaseVectorExtractor):
    """
    Extract steering vectors using My Custom Method.

    Description of your method and algorithm here.
    """

    def __init__(self):
        self.logger = get_logger("my_method_extractor")

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract steering vectors from specified layers.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths to extract from
            dataset_path: Path to dataset
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping layer names to steering vectors
        """
        self.logger.info(f"Extracting with MyMethod from {len(layers)} layers")

        # Your extraction logic here
        steering_vectors = {}

        for layer_path in layers:
            # 1. Access the layer
            # 2. Collect activations
            # 3. Compute steering vector
            # 4. Store in dictionary
            pass

        return steering_vectors
```

### 2. Register Extractor

Update `src/psyctl/core/steering_extractor.py` to register your method:

```python
from psyctl.core.extractors.my_method_extractor import MyMethodExtractor

class SteeringExtractor:
    EXTRACTORS = {
        'mean_contrastive': MeanContrastiveActivationVectorExtractor,
        'bipo': BiPOVectorExtractor,
        'my_method': MyMethodExtractor,  # Add your extractor
    }

    def extract(self, method: str = 'mean_contrastive', **kwargs):
        extractor_class = self.EXTRACTORS.get(method)
        if extractor_class is None:
            raise ValueError(f"Unknown extraction method: {method}")

        extractor = extractor_class()
        return extractor.extract(**kwargs)
```

### 3. Update CLI

Add method selection to CLI command in `src/psyctl/commands/extract.py`:

```python
@click.command()
@click.option("--model", required=True)
@click.option("--layer", multiple=True)
@click.option("--dataset", required=True, type=click.Path())
@click.option("--output", required=True, type=click.Path())
@click.option("--method", default="mean_contrastive",
              help="Extraction method: mean_contrastive, bipo, my_method")
@click.option("--lr", type=float, default=5e-4, help="Learning rate for BiPO")
@click.option("--beta", type=float, default=0.1, help="Beta parameter for BiPO")
@click.option("--epochs", type=int, default=10, help="Number of epochs for BiPO")
def steering(model: str, layer: tuple, dataset: str, output: str, method: str,
             lr: float, beta: float, epochs: int):
    # ...
    method_params = {}
    if method == "bipo":
        method_params = {"lr": lr, "beta": beta, "epochs": epochs}

    extractor.extract(method=method, **method_params)
```

### 4. Add Tests

Create tests in `tests/core/extractors/test_my_method_extractor.py`:

```python
import pytest
from psyctl.core.extractors.my_method_extractor import MyMethodExtractor


def test_my_method_basic():
    extractor = MyMethodExtractor()
    # Test basic functionality
    pass


def test_my_method_multi_layer():
    extractor = MyMethodExtractor()
    # Test multi-layer extraction
    pass
```

### 5. Document Your Method

Add documentation to this file under [Extraction Methods](#extraction-methods):

```markdown
### MyMethodName

Brief description of the method.

**Algorithm:**
1. Step 1
2. Step 2
3. Step 3

**Key Features:**
- Feature 1
- Feature 2

**When to use:**
- Use case 1
- Use case 2

**Parameters:**
- `param1`: Description
- `param2`: Description
```

## Implementation Details

### Layer Access

The `LayerAccessor` class handles dynamic layer access:

```python
from psyctl.core.layer_accessor import LayerAccessor

accessor = LayerAccessor()
layer_module = accessor.get_layer(model, "model.layers[13].mlp.down_proj")
```

### Activation Collection

The `ActivationHookManager` manages forward hooks:

```python
from psyctl.core.hook_manager import ActivationHookManager

hook_manager = ActivationHookManager()
layer_modules = {"layer_13": model.model.layers[13].mlp.down_proj}
hook_manager.register_hooks(layer_modules)

# Run inference
with torch.inference_mode():
    outputs = model(**inputs)

# Get collected activations
activations = hook_manager.get_mean_activations()
hook_manager.remove_all_hooks()
```

### Dataset Format

CAA datasets are JSONL files with this structure:

```json
{
  "question": "[Situation]\n...\n[Question]\n...\n1. Answer option 1\n2. Answer option 2\n[Answer]",
  "positive": "(1",
  "neutral": "(2"
}
```

The loader automatically combines `question` with `positive`/`neutral` to create full prompts.

## Troubleshooting

### Layer Not Found

```
Error: Layer 'model.layers[50].mlp.down_proj' not found in model
```

**Solution:** Check model architecture and available layers:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name")
print(model)  # Inspect structure
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size:

```bash
psyctl extract.steering ... --batch-size 8
```

Or set environment variable:

```bash
export PSYCTL_INFERENCE_BATCH_SIZE=8
```

### Token Position Issues

If activations seem incorrect, verify token position detection for your model architecture. Check logs for detected position or implement custom detection logic.

## References

- [CAA Paper: Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [BiPO Paper: Bi-Directional Preference Optimization](https://arxiv.org/abs/2406.00045)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [PSYCTL Dataset Building](./DATASET.BUILD.CAA.md)
- [PSYCTL Steering Application](./STEERING.md)
