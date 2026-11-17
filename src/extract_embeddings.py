
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


MODEL_ID = "openai/clip-vit-base-patch32"

IMAGE_DIR = Path("data/raw/bold5000/stimuli/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli")
OUTPUT_DIR = Path("data/processed/embeddings")

# Use batch to make good use of the GPU, also prevents loading all the images in memory
BATCH_SIZE = 16
# All the image in our dataset are .jpg, this is for robustness
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# I ran the program on a macbook, might be different for you, this assesses whether we can use GPU
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    # Find all stimulus images
    image_paths = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    print(f"Found {len(image_paths)} images.")

    # Load the pretrained CLIP model
    device = get_device()
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    # Don't wanna train CLIP, just use the existing pretrained model
    model.eval()

    # Process images in batches
    embedding_batches = []
    image_index = []

    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]

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

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.inference_mode():
            vision_output = model.vision_model(pixel_values=pixel_values)

            features = model.visual_projection(
                vision_output.pooler_output
            )

            features = F.normalize(features, p=2, dim=1)

        embedding_batches.append(features.cpu().numpy())

        # Keep the exact correspondence between rows and images.
        for path in valid_paths:
            image_index.append({
                "embedding_index": len(image_index),
                "filename": path.name,
                "relative_path": path.relative_to(IMAGE_DIR).as_posix(),
            })

        print(f"Processed {min(start + BATCH_SIZE, len(image_paths))}/{len(image_paths)}")

    # Combine all batches into one matrix.
    if not embedding_batches:
        raise RuntimeError("No images could be processed.")

    embeddings = np.concatenate(embedding_batches, axis=0)

    # Save embeddings and their corresponding image index.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "clip_vit_b32.npy", embeddings)

    index_path = OUTPUT_DIR / "clip_vit_b32_index.csv"

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["embedding_index", "filename", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(image_index)

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved embeddings to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
