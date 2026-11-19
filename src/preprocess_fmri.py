"""Build a trial-aligned fMRI matrix from the BOLD5000 ROI betas.

Output (per subject):
    data/processed/fmri/<subject>_roi.npy      (n_trials, n_voxels) float32, z-scored per session
    data/processed/fmri/<subject>_trials.csv   one row per trial, same order as the matrix
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = PROJECT_ROOT / "data/raw/BOLD5000_GLMsingle_ROI_betas/py"
# The stimuli archive is extracted either directly under bold5000/ or under bold5000/stimuli/.
LIST_DIR = next(
    (path for path in [
        PROJECT_ROOT / "data/raw/bold5000/BOLD5000_Stimuli/Stimuli_Presentation_Lists",
        PROJECT_ROOT / "data/raw/bold5000/stimuli/BOLD5000_Stimuli/Stimuli_Presentation_Lists",
    ] if path.is_dir()),
    None,
)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/fmri"

# Fixed ROI order so voxel columns mean the same thing on every run.
# CSI2/CSI3 files name LOC as "LO" (and one file is spelled "RRSC"), hence the patterns.
ROI_PATTERNS = {
    "LHEarlyVis": r"_LHEarlyVis\.npy$", "RHEarlyVis": r"_RHEarlyVis\.npy$",
    "LHLOC": r"_LHLOC?\.npy$", "RHLOC": r"_RHLOC?\.npy$",
    "LHOPA": r"_LHOPA\.npy$", "RHOPA": r"_RHOPA\.npy$",
    "LHPPA": r"_LHPPA\.npy$", "RHPPA": r"_RHPPA\.npy$",
    "LHRSC": r"_LHR?RSC\.npy$", "RHRSC": r"_RHR?RSC\.npy$",
}


def load_roi_betas(subject):
    """Concatenate the 10 ROI files along the voxel axis."""
    files = sorted(ROI_DIR.glob(f"{subject}_*.npy"))
    blocks = []
    for roi, pattern in ROI_PATTERNS.items():
        matches = [f for f in files if re.search(pattern, f.name)]
        if len(matches) != 1:
            raise FileNotFoundError(f"{subject} {roi}: expected 1 file, found {matches}")
        blocks.append(np.load(matches[0]))
        print(f"  {roi:<11} {blocks[-1].shape[1]:>4} voxels")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def load_trials(subject):
    """Trial table from the official presentation lists.

    Rows are ordered by (session, run, trial) using the numbers in the file names.
    The ROI betas follow this order: repeated presentations of the same image
    correlate (~0.10) only under this ordering, and ~0 under a shuffled one.
    """
    if LIST_DIR is None:
        raise FileNotFoundError("Stimuli_Presentation_Lists not found under data/raw/bold5000/")
    rows = []
    for path in (LIST_DIR / subject).glob("*/*.txt"):
        session, run = map(int, re.search(r"sess(\d+)_run(\d+)", path.name).groups())
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        for trial, name in enumerate(lines, start=1):
            rows.append({
                "session": session,
                "run": run,
                "trial": trial,
                "filename": name.removeprefix("rep_"),
                "is_rep": name.startswith("rep_"),
            })
    trials = pd.DataFrame(rows).sort_values(["session", "run", "trial"])
    return trials.reset_index(drop=True)


def zscore_per_session(X, sessions):
    """Standardise each voxel within each session.

    Scanner drift and head position change between sessions, so the same voxel can
    have a different baseline/scale on different days. Removing the per-session mean
    and std keeps only the trial-to-trial variation within a session.
    """
    X = X.copy()
    for session in np.unique(sessions):
        mask = sessions == session
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        X[mask] = (X[mask] - mean) / np.where(std > 0, std, 1.0)
    return X


def repeat_correlation(X, filenames):
    """Mean correlation between trials that showed the same image.

    If trials and beta rows are aligned, repeats of an image should respond more
    alike than random trial pairs (whose expected correlation is ~0).
    """
    Z = X - X.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    rows_by_image = pd.Series(np.arange(len(filenames))).groupby(np.asarray(filenames))
    correlations = [
        Z[a] @ Z[b]
        for rows in rows_by_image.groups.values() if len(rows) > 1
        for i, a in enumerate(rows) for b in rows[i + 1:]
    ]
    return float(np.mean(correlations))


def main():
    parser = argparse.ArgumentParser(description="Preprocess BOLD5000 ROI betas.")
    parser.add_argument("--subject", default="CSI1", choices=["CSI1", "CSI2", "CSI3", "CSI4"])
    args = parser.parse_args()

    print(f"{args.subject}: loading ROI betas")
    X = load_roi_betas(args.subject)
    trials = load_trials(args.subject)
    if len(trials) != X.shape[0]:
        raise AssertionError(f"{len(trials)} listed trials but {X.shape[0]} beta rows")

    X = zscore_per_session(X, trials["session"].to_numpy())

    shuffled = np.random.default_rng(0).permutation(trials["filename"].to_numpy())
    aligned_r = repeat_correlation(X, trials["filename"].to_numpy())
    shuffled_r = repeat_correlation(X, shuffled)
    print(f"Repeat correlation: {aligned_r:.3f} (shuffled order: {shuffled_r:.3f})")
    if aligned_r < 0.03:
        raise AssertionError("Repeated images do not correlate: trial alignment looks wrong")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{args.subject}_roi.npy", X)
    trials.to_csv(OUTPUT_DIR / f"{args.subject}_trials.csv", index_label="row")
    print(f"Saved {X.shape} matrix and {len(trials)} trials "
          f"({trials['filename'].nunique()} unique images) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
