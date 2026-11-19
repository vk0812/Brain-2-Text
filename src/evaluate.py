"""Retrieval metrics: rank a gallery of image embeddings for each predicted embedding."""

import numpy as np


def retrieval_ranks(predicted, gallery, target_index):
    """Rank (1 = best) of the correct gallery item for every query.

    predicted:    (n_queries, d) predicted embeddings
    gallery:      (n_gallery, d) candidate image embeddings
    target_index: (n_queries,)   row in `gallery` of the image actually shown
    """
    # Cosine similarity = dot product of L2-normalised vectors.
    predicted = predicted / np.linalg.norm(predicted, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    similarity = predicted @ gallery.T

    correct = similarity[np.arange(len(similarity)), target_index]
    # Rank = 1 + number of gallery items scored strictly higher than the correct one.
    return 1 + (similarity > correct[:, None]).sum(axis=1)


def retrieval_metrics(ranks, gallery_size):
    return {
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
        "gallery_size": int(gallery_size),
        "chance_recall@1": 1 / gallery_size,
        "chance_recall@5": 5 / gallery_size,
        "chance_median_rank": (gallery_size + 1) / 2,
    }
