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
# The stimuli archive is extracted either directly under bold5000/ or under bold5000/stimuli/.
_IMAGE_DIRS = [
    PROJECT_ROOT / "data/raw/bold5000" / sub / "BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli"
    for sub in ("", "stimuli")
]
IMAGE_DIR = next((path for path in _IMAGE_DIRS if path.is_dir()), _IMAGE_DIRS[0])
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
    "clip_vit_l14": {
        "model_id": "openai/clip-vit-large-patch14",
        "output_stem": "clip_vit_l14",
        "architecture": "clip",
    },
    # CLIP's own ResNet-50 image tower: same training objective as the CLIP ViTs,
    # same backbone family as the ImageNet ResNet-50 below.
    "clip_rn50": {
        "model_id": "RN50",
        "pretrained": "openai",
        "output_stem": "clip_rn50",
        "architecture": "open_clip",
    },
    # ImageNet-supervised CNNs from torchvision; "feature" names the layer we keep.
    "resnet50": {
        "model_id": "resnet50",
        "weights": "ResNet50_Weights.IMAGENET1K_V1",
        "feature": "avgpool",
        "output_stem": "resnet50",
        "architecture": "torchvision",
    },
    "convnext_base": {
        "model_id": "convnext_base",
        "weights": "ConvNeXt_Base_Weights.IMAGENET1K_V1",
        "feature": "avgpool + layernorm",
        "output_stem": "convnext_base",
        "architecture": "torchvision",
    },
    "alexnet_fc7": {
        "model_id": "alexnet",
        "weights": "AlexNet_Weights.IMAGENET1K_V1",
        "feature": "fc7 (after ReLU)",
        "output_stem": "alexnet_fc7",
        "architecture": "torchvision",
    },
}


def _load_torchvision(config):
    """Load a torchvision classifier with its head cut off at the configured layer."""
    import torchvision.models as tv_models

    enum_name, member = config["weights"].split(".")
    weights = getattr(getattr(tv_models, enum_name), member)
    model = getattr(tv_models, config["model_id"])(weights=weights)
    name = config["model_id"]
    if name == "resnet50":
        model.fc = torch.nn.Identity()  # 2048-d global-average-pooled features
    elif name == "convnext_base":
        model.classifier = model.classifier[:2]  # LayerNorm2d + Flatten -> 1024-d
    elif name == "alexnet":
        model.classifier = model.classifier[:6]  # up to fc7 + ReLU -> 4096-d
    else:
        raise ValueError(f"No feature layer defined for torchvision model {name!r}")
    return weights.transforms(), model


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
    elif config["architecture"] == "open_clip":
        import open_clip

        model, _, processor = open_clip.create_model_and_transforms(
            model_id, pretrained=config["pretrained"]
        )
    elif config["architecture"] == "torchvision":
        processor, model = _load_torchvision(config)
    else:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

    return processor, model.to(device).eval()


def encode_images(images, processor, model, model_key, device):
    """Encode RGB PIL images and return unit-normalized feature vectors."""
    architecture = MODEL_CONFIGS[model_key]["architecture"]
    if architecture in ("open_clip", "torchvision"):
        # These preprocessors are plain per-image transforms (PIL -> tensor).
        pixel_values = torch.stack([processor(image) for image in images]).to(device)
    else:
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        if architecture == "open_clip":
            features = model.encode_image(pixel_values)
        elif architecture == "torchvision":
            features = model(pixel_values)
        elif architecture == "clip":
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
        description="Extract image embeddings with one or all configured encoders."
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
