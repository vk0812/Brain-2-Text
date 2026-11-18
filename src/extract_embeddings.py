"""Extract normalized image embeddings from one or more pretrained vision models."""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = (
    PROJECT_ROOT
    / "data/raw/bold5000/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/embeddings"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Keeping the model-specific details here makes it straightforward to add another
# image encoder without changing the extraction and saving pipeline.
MODEL_CONFIGS = {
    "clip_vit_b32": {
        "model_id": "openai/clip-vit-base-patch32",
        "output_stem": "clip_vit_b32",
        "architecture": "clip",
    },
    "dinov2_large": {
        "model_id": "facebook/dinov2-large",
        "output_stem": "dinov2_large",
        "architecture": "dinov2",
    },
}


def get_device():
    """Prefer an available accelerator, falling back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_images(image_dir):
    """Return all supported image files in deterministic order."""
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_encoder(model_key, device):
    """Load a configured model and its image processor."""
    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]

    if config["architecture"] == "clip":
        processor = AutoProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
    else:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

    return processor, model.to(device).eval()


def encode_images(images, processor, model, model_key, device):
    """Encode RGB PIL images and return unit-normalized feature vectors."""
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        if MODEL_CONFIGS[model_key]["architecture"] == "clip":
            vision_output = model.vision_model(pixel_values=pixel_values)
            features = model.visual_projection(vision_output.pooler_output)
        else:
            # DINOv2's CLS token is its global image representation.
            vision_output = model(pixel_values=pixel_values)
            features = vision_output.last_hidden_state[:, 0, :]

        features = F.normalize(features.float(), p=2, dim=1)

    return features.cpu().numpy()


def extract_model(
    model_key,
    image_dir=IMAGE_DIR,
    output_dir=OUTPUT_DIR,
    batch_size=8,
    limit=None,
    device=None,
):
    """Extract embeddings and save them with a row-aligned CSV image index."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_key!r}; choose from {list(MODEL_CONFIGS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    image_paths = find_images(image_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    device = device or get_device()
    print(f"{model_key}: found {len(image_paths)} images; using {device}")
    processor, model = load_encoder(model_key, device)

    embedding_batches = []
    image_index = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        valid_paths = []

        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
                valid_paths.append(path)
            except (OSError, ValueError) as error:
                print(f"Skipping {path}: {error}")

        if not images:
            continue

        embedding_batches.append(
            encode_images(images, processor, model, model_key, device)
        )

        for path in valid_paths:
            image_index.append(
                {
                    "embedding_index": len(image_index),
                    "filename": path.name,
                    "relative_path": path.relative_to(image_dir).as_posix(),
                }
            )

        print(f"Processed {min(start + batch_size, len(image_paths))}/{len(image_paths)}")

    if not embedding_batches:
        raise RuntimeError("No images could be processed.")

    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / f"{MODEL_CONFIGS[model_key]['output_stem']}.npy"
    index_path = output_dir / f"{MODEL_CONFIGS[model_key]['output_stem']}_index.csv"

    np.save(embedding_path, embeddings)
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["embedding_index", "filename", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(image_index)

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved embeddings to {embedding_path}")
    print(f"Saved image index to {index_path}")

    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return embedding_path, index_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image embeddings using CLIP, DINOv2, or both."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CONFIGS, "all"],
        default="clip_vit_b32",
        help="Encoder to run (default: clip_vit_b32).",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optionally process only the first N sorted images (useful for smoke tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for model_key in model_keys:
        extract_model(
            model_key,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
