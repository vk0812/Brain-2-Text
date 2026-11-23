"""Map ROI fMRI betas to image embeddings and evaluate by retrieval.

Models: ridge (linear baseline), mlp (cosine loss) and mlp_infonce (contrastive
loss). All follow the same protocol:
choose hyperparameters on validation images, refit on train + val, then
evaluate once on the test images.

Run src/preprocess_fmri.py first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluate import retrieval_metrics, retrieval_ranks
from models import predict_mlp, train_mlp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FMRI_DIR = PROJECT_ROOT / "data/processed/fmri"
EMBEDDING_DIR = PROJECT_ROOT / "data/processed/embeddings"
RESULTS_DIR = PROJECT_ROOT / "outputs/results"

# These two files are byte-identical. Treat them as one image so the same picture
# can't land in both train and test, and isn't counted as a "wrong" retrieval.
DUPLICATE_IMAGES = {"n03447721_2079.JPEG": "n03017168_3695.JPEG"}

RIDGE_ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
MLP_WEIGHT_DECAYS = [1e-2, 1e-1, 1.0]
MLP_MAX_EPOCHS = 60
# Weight decay barely mattered for the cosine MLP, so the contrastive MLP fixes it
# and tunes the temperature instead.
INFONCE_TEMPERATURES = [0.01, 0.03, 0.1]
INFONCE_WEIGHT_DECAY = 0.1


def split_images(image_ids, test_fraction, val_fraction, seed):
    """Split unique image identities (not trials) into train / val / test."""
    rng = np.random.default_rng(seed)
    images = rng.permutation(sorted(set(image_ids)))
    n_test = round(len(images) * test_fraction)
    n_val = round(len(images) * val_fraction)
    return (
        set(images[n_test + n_val:]),
        set(images[n_test:n_test + n_val]),
        set(images[:n_test]),
    )


def evaluate_split(predicted, image_ids, embeddings, row_of):
    """Retrieve among the images of this split only; each trial is one query."""
    gallery_images = sorted(set(image_ids))
    gallery = embeddings[[row_of[name] for name in gallery_images]]
    position = {name: i for i, name in enumerate(gallery_images)}
    target = np.array([position[name] for name in image_ids])
    ranks = retrieval_ranks(predicted, gallery, target)
    return retrieval_metrics(ranks, len(gallery_images))


def fit_ridge(X, Y, is_train, is_val, val_score, seed, labels):
    """Pick alpha on val, refit on train + val. Returns (predict function, chosen params)."""
    best_alpha, best_rank = None, np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(X[is_train], Y[is_train])
        rank = val_score(model.predict(X[is_val]), f"alpha={alpha:g}")
        if rank < best_rank:
            best_alpha, best_rank = alpha, rank

    fit_mask = is_train | is_val
    model = Ridge(alpha=best_alpha).fit(X[fit_mask], Y[fit_mask])
    return model.predict, {"alpha": best_alpha}


def fit_mlp(X, Y, is_train, is_val, val_score, seed, labels):
    """Pick weight decay and number of epochs on val, refit on train + val."""
    best = {"weight_decay": None, "epochs": None, "rank": np.inf}
    for weight_decay in MLP_WEIGHT_DECAYS:
        def track(epoch, model):
            # Score every 5 epochs: a cheap form of early stopping.
            if epoch % 5:
                return
            rank = val_score(predict_mlp(model, X[is_val]),
                             f"wd={weight_decay:g} epoch={epoch}", verbose=epoch % 20 == 0)
            if rank < best["rank"]:
                best.update(weight_decay=weight_decay, epochs=epoch, rank=rank)

        train_mlp(X[is_train], Y[is_train], MLP_MAX_EPOCHS, weight_decay, seed,
                  on_epoch_end=track)

    fit_mask = is_train | is_val
    model = train_mlp(X[fit_mask], Y[fit_mask], best["epochs"], best["weight_decay"], seed)
    return (lambda X_new: predict_mlp(model, X_new),
            {"weight_decay": best["weight_decay"], "epochs": best["epochs"]})


def fit_mlp_infonce(X, Y, is_train, is_val, val_score, seed, labels):
    """Contrastive MLP: pick temperature and number of epochs on val, refit on train + val."""
    best = {"temperature": None, "epochs": None, "rank": np.inf}
    for temperature in INFONCE_TEMPERATURES:
        def track(epoch, model):
            if epoch % 5:
                return
            rank = val_score(predict_mlp(model, X[is_val]),
                             f"tau={temperature:g} epoch={epoch}", verbose=epoch % 20 == 0)
            if rank < best["rank"]:
                best.update(temperature=temperature, epochs=epoch, rank=rank)

        train_mlp(X[is_train], Y[is_train], MLP_MAX_EPOCHS, INFONCE_WEIGHT_DECAY, seed,
                  on_epoch_end=track, loss="infonce", labels=labels[is_train],
                  temperature=temperature)

    fit_mask = is_train | is_val
    model = train_mlp(X[fit_mask], Y[fit_mask], best["epochs"], INFONCE_WEIGHT_DECAY, seed,
                      loss="infonce", labels=labels[fit_mask], temperature=best["temperature"])
    return (lambda X_new: predict_mlp(model, X_new),
            {"temperature": best["temperature"], "epochs": best["epochs"]})


FIT_FUNCTIONS = {"ridge": fit_ridge, "mlp": fit_mlp, "mlp_infonce": fit_mlp_infonce}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an fMRI → embedding model.")
    parser.add_argument("--model", default="ridge", choices=list(FIT_FUNCTIONS))
    parser.add_argument("--subject", default="CSI1")
    parser.add_argument("--embedding", default="clip_vit_b32",
                        choices=sorted(path.stem for path in EMBEDDING_DIR.glob("*.npy")))
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # fMRI trials (rows of X) and the image shown on each trial.
    X = np.load(FMRI_DIR / f"{args.subject}_roi.npy")
    trials = pd.read_csv(FMRI_DIR / f"{args.subject}_trials.csv")
    image_ids = trials["filename"].replace(DUPLICATE_IMAGES).to_numpy()

    # Image embeddings, looked up by filename (never by position).
    embeddings = np.load(EMBEDDING_DIR / f"{args.embedding}.npy")
    index = pd.read_csv(EMBEDDING_DIR / f"{args.embedding}_index.csv")
    row_of = dict(zip(index["filename"], index["embedding_index"]))
    missing = set(image_ids) - set(row_of)
    if missing:
        raise KeyError(f"{len(missing)} trial images have no embedding, e.g. {list(missing)[:3]}")
    Y = embeddings[[row_of[name] for name in image_ids]]

    train_images, val_images, test_images = split_images(
        image_ids, args.test_fraction, args.val_fraction, args.seed
    )
    is_train = np.isin(image_ids, list(train_images))
    is_val = np.isin(image_ids, list(val_images))
    is_test = np.isin(image_ids, list(test_images))
    print(f"{args.model} | {args.subject} / {args.embedding}: X {X.shape}, Y {Y.shape}")
    print(f"Images  train {len(train_images)}  val {len(val_images)}  test {len(test_images)}")
    print(f"Trials  train {is_train.sum()}  val {is_val.sum()}  test {is_test.sum()}")

    def val_score(predicted, label, verbose=True):
        metrics = evaluate_split(predicted, image_ids[is_val], embeddings, row_of)
        if verbose:
            print(f"  {label:<22} val R@1={metrics['recall@1']:.3f}  "
                  f"R@5={metrics['recall@5']:.3f}  median rank={metrics['median_rank']:.0f}")
        return metrics["median_rank"]

    # One integer per image identity: the contrastive loss uses it to avoid treating
    # two presentations of the same image as negatives of each other.
    labels = pd.factorize(image_ids)[0]
    predict, params = FIT_FUNCTIONS[args.model](X, Y, is_train, is_val, val_score,
                                                args.seed, labels)
    test = evaluate_split(predict(X[is_test]), image_ids[is_test], embeddings, row_of)

    print(f"\nTest ({params}, {test['gallery_size']} candidate images)")
    print(f"  Recall@1     {test['recall@1']:.3f}   (chance {test['chance_recall@1']:.3f})")
    print(f"  Recall@5     {test['recall@5']:.3f}   (chance {test['chance_recall@5']:.3f})")
    print(f"  Median rank  {test['median_rank']:.0f}   (chance {test['chance_median_rank']:.0f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{args.model}_{args.subject}_{args.embedding}_seed{args.seed}.json"
    result_path.write_text(json.dumps({**vars(args), **params, "test": test}, indent=2))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
