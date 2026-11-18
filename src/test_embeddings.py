"""Validate saved embedding files and re-infer random images against their rows."""

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from extract_embeddings import (
    IMAGE_DIR,
    IMAGE_EXTENSIONS,
    MODEL_CONFIGS,
    OUTPUT_DIR,
    encode_images,
    find_images,
    get_device,
    load_encoder,
)


def load_index(index_path):
    with index_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def verify_model(model_key, image_dir, output_dir, sample_count, seed, threshold):
    stem = MODEL_CONFIGS[model_key]["output_stem"]
    embedding_path = output_dir / f"{stem}.npy"
    index_path = output_dir / f"{stem}_index.csv"
    if not embedding_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Missing files for {model_key}: expected {embedding_path} and {index_path}"
        )

    embeddings = np.load(embedding_path, mmap_mode="r")
    rows = load_index(index_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise AssertionError(
            f"{model_key}: embedding shape {embeddings.shape} does not match "
            f"{len(rows)} index rows"
        )
    image_paths = find_images(image_dir)
    expected_relative_paths = [
        path.relative_to(image_dir).as_posix() for path in image_paths
    ]
    mapped_relative_paths = [row["relative_path"] for row in rows]
    if mapped_relative_paths != expected_relative_paths:
        raise AssertionError(
            f"{model_key}: mappings do not exactly match the sorted image list "
            f"under {image_dir} ({len(rows)} mappings, {len(image_paths)} images)"
        )

    for row_number, row in enumerate(rows):
        if int(row["embedding_index"]) != row_number:
            raise AssertionError(f"{model_key}: nonsequential index at row {row_number}")
        image_path = image_dir / row["relative_path"]
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise AssertionError(f"{model_key}: missing or invalid image path {image_path}")
        if row["filename"] != image_path.name:
            raise AssertionError(
                f"{model_key}: filename/path mismatch at row {row_number}: "
                f"{row['filename']} != {image_path.name}"
            )

    # Check the entire stored matrix cheaply before loading the vision model.
    if not np.isfinite(embeddings).all():
        raise AssertionError(f"{model_key}: embeddings contain NaN or infinity")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise AssertionError(f"{model_key}: embeddings are not L2-normalized")

    rng = np.random.default_rng(seed)
    selected_rows = np.sort(
        rng.choice(len(rows), size=min(sample_count, len(rows)), replace=False)
    )
    selected_paths = [image_dir / rows[index]["relative_path"] for index in selected_rows]
    images = []
    for path in selected_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    device = get_device()
    print(f"{model_key}: re-inferring {len(selected_paths)} sampled images on {device}")
    processor, model = load_encoder(model_key, device)
    inferred = encode_images(images, processor, model, model_key, device)
    stored = np.asarray(embeddings[selected_rows])
    similarities = np.sum(inferred * stored, axis=1)
    minimum_similarity = float(similarities.min())
    if minimum_similarity < threshold:
        worst = int(np.argmin(similarities))
        raise AssertionError(
            f"{model_key}: sampled embedding mismatch for "
            f"{rows[selected_rows[worst]]['relative_path']} "
            f"(cosine similarity {minimum_similarity:.6f} < {threshold})"
        )

    random_sample = int(rng.integers(len(selected_rows)))
    sample_row = int(selected_rows[random_sample])
    print(
        f"{model_key}: PASS; shape={embeddings.shape}; "
        f"min sampled cosine={minimum_similarity:.6f}; "
        f"random image={rows[sample_row]['relative_path']} "
        f"(cosine={similarities[random_sample]:.6f})"
    )
    del model, processor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check saved embeddings, mappings, and sampled image/model agreement."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CONFIGS, "all"],
        default="all",
        help="Model outputs to verify (default: all).",
    )
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    model_keys = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for model_key in model_keys:
        verify_model(
            model_key,
            args.image_dir,
            args.output_dir,
            args.samples,
            args.seed,
            args.cosine_threshold,
        )


if __name__ == "__main__":
    main()
