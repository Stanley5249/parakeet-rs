# melops-export

Export NVIDIA Parakeet-TDT ASR models to ONNX format.

Supports NeMo EncDecRNNTBPE models (Parakeet-TDT family).

## Installation

```bash
pixi install -e export
```

## Usage

```bash
# Export default model (nvidia/parakeet-tdt-0.6b-v3)
pixi run export

# Export specific model
pixi run export --repo-id nvidia/parakeet-tdt-0.6b-v3 --out-dir models/parakeet

# Export from local .nemo file
pixi run export --repo-id /path/to/model.nemo --out-dir models/custom
```

### Python API

```python
from melops_export import export_model

export_model()  # Default model
export_model(repo_id="nvidia/parakeet-tdt-0.6b-v3", out_dir="models/parakeet")
export_model(repo_id="/path/to/model.nemo", out_dir="models/custom")
```

## Output

Exports 5 files per model to `.cache/melops/models/<repo_id>/`:

| File                       | Size (0.6b) | Description              |
| -------------------------- | ----------- | ------------------------ |
| `encoder-model.onnx`       | ~39.9MB     | Encoder graph            |
| `encoder-model.data`       | ~2.3GB      | Encoder weight           |
| `decoder_joint-model.onnx` | ~7.3KB      | Decoder+joint graph      |
| `decoder_joint-model.data` | ~69.1MB     | Decoder+joint weights    |
| `tokenizer.json`           | ~173KB      | SentencePiece vocabulary |

Total: ~2.37GB for 0.6b model.

**Note:** NeMo exports create 295+ weight files; this script consolidates them into single `.data` files. Both `.onnx` and `.data` files are required for inference.
