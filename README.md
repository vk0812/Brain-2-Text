# Brain-2-Text

## Repository structure

```text
.
├── Cheatsheets/
│   ├── BOLD5000.md
│   ├── Concepts.md
│   └── assets/
├── data/
│   ├── raw/
│   │   ├── bold5000/BOLD5000_Stimuli/
│   │   │   ├── Image_Labels/
│   │   │   ├── Localizer_Stimuli/
│   │   │   ├── Scene_Stimuli/Presented_Stimuli/{COCO,ImageNet,Scene}/
│   │   │   └── Stimuli_Presentation_Lists/
│   │   ├── BOLD5000_Stimuli_samples/Presented_Stimuli/
│   │   ├── BOLD5000_metadata/
│   │   │   ├── CSI1_events/
│   │   │   └── CSI1_presentation_lists/
│   │   └── BOLD5000_GLMsingle_ROI_betas/{mat,py}/
│   └── processed/embeddings/
├── Notebooks/bold5000/data_exploration.ipynb
├── outputs/bold5000/CSI1_trial_image_manifest.csv
├── src/
├── pyproject.toml
├── uv.lock
└── PLAN.md
```

`data/raw/` contains local BOLD5000 stimulus images, metadata, and GLMsingle ROI beta files. The full `Presented_Stimuli` scene image directory currently contains **4,916 images**. `BOLD5000_Stimuli_samples/Presented_Stimuli/` is a small sample directory with four images.

The CSI1 trial image manifest in `outputs/bold5000/` has **5,254 presentation rows** (plus its header) and records trial identifiers, image names, repetition information, event-file references, and available image paths.

The notebook explores the BOLD5000 data and the relationship between CSI1 image presentations, event rows, and ROI response rows. The cheatsheets contain dataset notes and reference material for neuroscience and imaging concepts.

## Python environment

The project requires Python 3.12. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

```bash
uv sync
```

The embedding scripts use PyTorch, Transformers, NumPy, and Pillow. On first use, Transformers downloads the selected pretrained model from Hugging Face. Device selection prefers Apple MPS, then CUDA, then CPU.

## Source scripts

| File | Current contents |
| --- | --- |
| `src/extract_embeddings.py` | Configurable batched image embedding extraction for CLIP ViT-B/32 and DINOv2-Large. Saves an `.npy` feature matrix and a matching CSV image index. |
| `src/test_embeddings.py` | Checks saved matrices and mappings against the image directory, validates normalized finite vectors, and re-embeds a seeded random sample to check row alignment. |
| `src/preprocess_fmri.py` | Empty placeholder. |
| `src/models.py` | Empty placeholder. |
| `src/train.py` | Empty placeholder. |
| `src/evaluate.py` | Empty placeholder. |

## Image embeddings

The generated matrices are stored under `data/processed/embeddings/`. Each matrix is `float32`, L2-normalized, and ordered according to its corresponding index CSV. The index has one row per embedding, with columns `embedding_index`, `filename`, and `relative_path`.

| Model key | Hugging Face model | Embedding file | Shape | Index file | Approx. size |
| --- | --- | --- | --- | --- | --- |
| `clip_vit_b32` | `openai/clip-vit-base-patch32` | `clip_vit_b32.npy` | `(4916, 512)` | `clip_vit_b32_index.csv` | 9.6 MB + 291 KB |
| `dinov2_large` | `facebook/dinov2-large` | `dinov2_large.npy` | `(4916, 1024)` | `dinov2_large_index.csv` | 19 MB + 291 KB |

CLIP embeddings use the vision model's pooled output followed by CLIP's visual projection. DINOv2 embeddings use the final hidden state's CLS token. Both are L2-normalized. The CSV paths are relative to the presented-stimuli image directory.

Load a matrix and its row mapping with:

```python
import csv
import numpy as np

embeddings = np.load("data/processed/embeddings/dinov2_large.npy")
with open(
    "data/processed/embeddings/dinov2_large_index.csv",
    newline="",
    encoding="utf-8",
) as file:
    image_index = list(csv.DictReader(file))

# embeddings[i] corresponds to image_index[i]["relative_path"]
```

## Running embedding scripts

Extract embeddings for one model:

```bash
uv run python src/extract_embeddings.py --model clip_vit_b32
uv run python src/extract_embeddings.py --model dinov2_large
```

Run both models:

```bash
uv run python src/extract_embeddings.py --model all
```

The default model is `clip_vit_b32`. `--batch-size` controls inference batch size; `--image-dir` and `--output-dir` override the default paths. `--limit N` processes only the first N images in sorted path order. Use a separate output directory for a limited smoke run so it does not replace the full-dataset outputs:

```bash
uv run python src/extract_embeddings.py \
  --model dinov2_large \
  --limit 8 \
  --output-dir outputs/embedding_smoke_test
```

Verify both full-dataset embedding stores and re-infer eight deterministic random images per model:

```bash
uv run python src/test_embeddings.py --model all --samples 8 --seed 42
```

The verifier checks that each index exactly matches the sorted image list, row indices are sequential, embeddings are finite and normalized, and sampled re-inference agrees with the saved vectors.

## Project Goals and Plans
