# Ablating Archetypes — Reproduction Code

This repository contains the minimal reproduction code for the paper:

> **Ablating Archetypes: The Stability of Archetypal SAEs is an Artifact of Initialization and Metric Design**

## Notebooks

There are two notebooks, differing only in whether activations are centered before k-means:

### `Archetypal_SAE_ablation.ipynb` — uncentered (close to the original Colab)

Replicates the preprocessing of the original Archetypal SAE Colab notebook (no centering). This is the most direct comparison to the authors' reference implementation. Because activations are not centered, the cosine stability metric is subject to the galaxy-far-away confound described in the paper (§3.2): all k-means centroids cluster near the activation mean, inflating cosine similarity for any method whose atoms stay close to that mean. The archetypal SAE benefits from this inflation, so its reported stability numbers are lower than in the centered version.

### `Archetypal_SAE_ablation_centered.ipynb` — centered (matches Table 3 in the paper)

Subtracts the activation mean before running k-means and initializes the decoder bias to that mean. This eliminates the cosine metric artifact. The numbers in this notebook match Table 3 of the paper. With centering applied, the ablated archetypal SAE and the classical SAE with k-means initialization both outperform the full archetypal SAE in stability, while also achieving better reconstruction (R²).

**Which to run?** Use the centered notebook to reproduce the paper's results. Use the uncentered notebook to see the confound in action: the archetypal SAE looks artificially stable precisely because its atoms stay near the uncentered data mean.

## What the ablations show

Both notebooks run the same four conditions:

| Condition | What it tests |
|---|---|
| Classic SAE (random init) | Baseline instability |
| Archetypal SAE (full) | Authors' reported method |
| Archetypal SAE (projection disabled) | Isolates k-means init from convex-hull constraint |
| Classic SAE (k-means init) | Confirms init is the source of stability |

Conditions 3 and 4 are structurally identical (same forward/backward pass, implemented two ways) and produce numerically identical results under fixed seeding. This confirms that the archetypal constraint itself is not responsible for the reported stability advantage.

## Setup

```bash
uv sync
```

Then open either notebook and run the `wget` cell to download the rabbit dataset. Requires a CUDA-capable GPU.

## Why the repo bundles a copy of `overcomplete`

The notebook depends on the [`overcomplete`](https://github.com/KempnerInstitute/Overcomplete) library. The standard pip-installable version loads all vision models eagerly on import, which causes the DinoV2 HuggingFace processor to conflict with some CUDA environments. This repo ships a stripped-down copy (only the files needed by the notebooks) with the unused model classes removed so that the import is lightweight and the DinoV2 loading path is clean.
