"""Brain-to-brain identification: how distinguishable are single-trial fMRI patterns?

For images shown several times, use one presentation as the query and try to
find the *same image's other presentation(s)* among those of all repeated
images, by correlation of voxel patterns. No image model is involved, so this
measures how much image-specific signal a single trial carries.

Two variants:
    single:   each gallery item is ONE other presentation (noise on both sides)
    averaged: each gallery item is the MEAN of 3 other presentations (less noise
              on the gallery side, closer to the decoder's situation, whose
              targets are noise-free image embeddings)

This is a reference point, not a strict upper bound: a trained decoder can learn
to weight informative voxels, while plain correlation weights all voxels equally.

Run src/preprocess_fmri.py first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import retrieval_metrics, retrieval_ranks
from train import DUPLICATE_IMAGES, FMRI_DIR, PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "outputs/noise_ceiling.json"
N_AVERAGED = 3


def center(patterns):
    """Subtract each pattern's mean so that cosine similarity = Pearson correlation."""
    return patterns - patterns.mean(axis=1, keepdims=True)


def brain_to_brain(X, image_ids, averaged):
    """Return the rank of the correct image for every presentation of a repeated image."""
    rows = pd.Series(np.arange(len(image_ids))).groupby(image_ids).apply(list)
    min_count = N_AVERAGED + 1 if averaged else 2
    rows = rows[rows.map(len) >= min_count]
    images = list(rows.index)

    ranks = []
    for i, image in enumerate(images):
        for query_row in rows[image]:
            gallery = []
            for other in images:
                # Never use the query presentation itself in the gallery.
                candidates = [r for r in rows[other] if r != query_row]
                if averaged:
                    gallery.append(X[candidates[:N_AVERAGED]].mean(axis=0))
                else:
                    gallery.append(X[candidates[0]])
            rank = retrieval_ranks(center(X[[query_row]]), center(np.stack(gallery)),
                                   np.array([i]))
            ranks.append(rank[0])
    return np.array(ranks), len(images)


def main():
    parser = argparse.ArgumentParser(description="Brain-to-brain identification ceiling.")
    parser.add_argument("--subjects", nargs="+", default=["CSI1", "CSI2", "CSI3", "CSI4"])
    args = parser.parse_args()

    results = {}
    print("| Subject | Variant | Gallery | Queries | Recall@1 (%) | Recall@5 (%) | Median rank | 2-way (%) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for subject in args.subjects:
        X = np.load(FMRI_DIR / f"{subject}_roi.npy")
        trials = pd.read_csv(FMRI_DIR / f"{subject}_trials.csv")
        image_ids = trials["filename"].replace(DUPLICATE_IMAGES).to_numpy()

        for variant in ["single", "averaged"]:
            ranks, n_images = brain_to_brain(X, image_ids, averaged=variant == "averaged")
            metrics = retrieval_metrics(ranks, n_images)
            metrics["queries"] = len(ranks)
            results[f"{subject}_{variant}"] = metrics
            print(f"| {subject} | {variant} | {n_images} | {len(ranks)} "
                  f"| {100 * metrics['recall@1']:.1f} | {100 * metrics['recall@5']:.1f} "
                  f"| {metrics['median_rank']:.0f} | {100 * metrics['two_way']:.1f} |")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
