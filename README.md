# Ablating Archetypes — Reproduction Code

This repository contains the minimal reproduction code for the paper:

> **Ablating Archetypes: The Stability of Archetypal SAEs is an Artifact of Initialization and Metric Design**

## What this is

`Archetypal_SAE_ablation.ipynb` is the core experiment notebook. It is built directly on the minimal working example provided by the authors of Archetypal SAEs in their public Colab notebook:

> https://colab.research.google.com/drive/1TmAtUhIdFGSMlDhKr2ndXGR8GU4R4aTq

We reproduce their setup (DinoV2 patch embeddings, rabbit image class, same hyperparameters) and add two ablation conditions that isolate the source of their reported stability advantage:

1. **Archetypal SAE with projection disabled** — retains k-means initialization, removes the convex-hull enforcement.
2. **Classical TopK SAE with fixed k-means initialization** — no archetypal constraint at any point during training.

Both conditions match the stability of the full Archetypal SAE, showing the advantage comes entirely from shared initialization rather than the archetypal constraint.

## Why the repo bundles a copy of `overcomplete`

The notebook depends on the [`overcomplete`](https://github.com/KempnerInstitute/overcomplete) library. The standard pip-installable version of `overcomplete` loads all vision models eagerly on import, which causes the DinoV2 HuggingFace processor to conflict with some CUDA environments. This repo ships a stripped-down copy of `overcomplete` (only the files needed by the notebook) with the unused model classes removed so that the import is lightweight and the DinoV2 loading path is clean.

## Setup

```bash
pip install -e .
```

Then open `Archetypal_SAE_ablation.ipynb` and download the rabbit dataset by running the `wget` cell.

Requires a CUDA-capable GPU.
