---
layout: post
title: Using DiffusionPDE as a Baseline on BubbleML (Incompressible Navier–Stokes)
date: 2026-03-09 12:00:00
description: A guide to which DiffusionPDE code files to use when running experiments on the BubbleML dataset governed by incompressible Navier–Stokes equations.
tags: research diffusion pde fluid-dynamics navier-stokes bubbleml sciml
categories: research-notes
related_posts: false
---

[DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE) {% cite huang2024diffusionpde %} is a generative framework for PDE-solving under partial observation. This note explains which files to use when adapting it as a baseline for the [BubbleML](https://github.com/HPCForge/BubbleML) dataset, whose dynamics are governed by the incompressible Navier–Stokes equations.

---

## Background

**BubbleML** simulates multiphase, multiphysics boiling processes. Each simulation stores four fields over time:

| Field | Symbol | Description                 |
| ----- | ------ | --------------------------- |
| `ux`  | $u$    | Horizontal velocity         |
| `uy`  | $v$    | Vertical velocity           |
| `T`   | $T$    | Temperature                 |
| `pf`  | $\phi$ | Liquid–vapor phase fraction |

The underlying physics are the **incompressible Navier–Stokes equations** in a bounded, wall-enclosed domain with a heated bottom surface — making the _bounded_ NS variant in DiffusionPDE the closest match.

**DiffusionPDE** supports two NS variants:

| Variant         | Domain                       | Config prefix   |
| --------------- | ---------------------------- | --------------- |
| `NS-NonBounded` | Periodic, no walls           | `ns-nonbounded` |
| `NS-Bounded`    | Cylinder obstacle in channel | `ns-bounded`    |

For BubbleML, start from the **`NS-Bounded`** variant because BubbleML has solid walls and non-periodic boundary conditions, unlike the periodic NS used in `NS-NonBounded`.

---

## Files to Reference

### 1. Configuration Files — `configs/`

These YAML files control data paths, the pre-trained model checkpoint, solver hyperparameters, and guidance weights. For BubbleML/NS, the relevant configs are:

| File                              | Purpose                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `configs/ns-bounded.yaml`         | Recover **both** coefficient (initial state) and solution jointly                  |
| `configs/ns-bounded-forward.yaml` | **Forward** problem: predict solution from sparse initial/coefficient observations |
| `configs/ns-bounded-inverse.yaml` | **Inverse** problem: recover initial state from sparse solution observations       |

> **Tip.** For a forecasting setting (given early time-steps, predict future states), the `forward` config is the natural starting point.

You will need to modify at least:

```yaml
data:
  name: "NS-Bounded"
  datapath: "data/testing/bubbleml/<your_sample>.npy" # path to your BubbleML data
  c_x: 41 # geometry params — adjust to BubbleML domain
  c_y: 63
  radius: 10
  offset: 0
```

---

### 2. Main Inference Script — `generate_pde.py`

Entry point for all PDE solving. It reads the config and dispatches to the correct solver:

```bash
python3 generate_pde.py --config configs/ns-bounded-forward.yaml
```

Internally, it calls `generate_ns_bounded(config)` from `scripts/`. When adapting for BubbleML, this is where you adjust the channel indexing to match BubbleML's `(ux, uy, T, pf)` layout.

---

### 3. NS Generation Script — `scripts/generate_ns_bounded.py`

This script implements the **diffusion-guided sampling loop** for the bounded NS case. Key sections to understand and modify for BubbleML:

- **Data loading**: replace DiffusionPDE's `.npy` loader with one that reads BubbleML's HDF5 format
- **Channel definitions**: DiffusionPDE uses a single velocity component `v`; BubbleML requires at minimum `(ux, uy)` and optionally `T` and `pf`
- **Boundary mask**: the cylinder mask (`c_x`, `c_y`, `radius`) must be replaced with BubbleML's flat heated-wall mask
- **PDE residual guidance** (`zeta_pde`): the NS residual term must be re-implemented for incompressible NS without a cylinder obstacle

---

### 4. Training Script — `train.py` + `merge_data.py`

To **train a new diffusion model** on BubbleML data instead of using the provided pretrained checkpoint:

```bash
# Step 1: Prepare merged .npy training files from BubbleML HDF5 data
# (write a custom converter — BubbleML uses .h5 / HDF5 format)
python3 merge_data.py  # or your custom conversion script

# Step 2: Train the diffusion model
torchrun --standalone --nproc_per_node=<N_GPUS> train.py \
    --outdir=pretrained-bubbleml \
    --data=data/bubbleml-merged/ \
    --cond=0 \
    --arch=ddpmpp \
    --batch=60 \
    --batch-gpu=20 \
    --tick=10 \
    --snap=50 \
    --dump=100 \
    --duration=20 \
    --ema=0.05
```

---

### 5. Data Generation Reference — `dataset_generation/`

DiffusionPDE's own NS data was generated using a 2-D fluid simulator. For BubbleML you use the real simulation dataset, but examining `dataset_generation/ns_bounded/` is useful for understanding:

- how the velocity fields are normalized to $[-1, 1]$
- the expected `.npy` shape `[N, X, Y, T]`
- how boundary conditions are encoded

---

### 6. Model Architecture — `training/`

The diffusion backbone is inherited from [EDM](https://github.com/NVlabs/edm). The relevant files are:

| File                      | Purpose                               |
| ------------------------- | ------------------------------------- |
| `training/networks.py`    | U-Net / DDPM++ architecture           |
| `training/loss.py`        | EDM preconditioning and training loss |
| `dnnlib/`, `torch_utils/` | EDM utilities (unchanged)             |

For BubbleML you may need to adjust the **input channel count** in `training/networks.py` if you add extra physical fields (`T`, `pf`) beyond the default two velocity components.

---

## Summary Checklist

| Step                             | File(s)                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| Choose problem type              | `configs/ns-bounded{,-forward,-inverse}.yaml`                  |
| Run inference                    | `generate_pde.py`                                              |
| Adapt data loading & masks       | `scripts/generate_ns_bounded.py`                               |
| Convert BubbleML → `.npy`        | Custom script (see `dataset_generation/` for format reference) |
| Train from scratch (optional)    | `train.py`, `merge_data.py`                                    |
| Modify input channels (optional) | `training/networks.py`                                         |

---

## Key Differences: BubbleML vs. DiffusionPDE's NS-Bounded

| Aspect      | DiffusionPDE NS-Bounded          | BubbleML                        |
| ----------- | -------------------------------- | ------------------------------- |
| Geometry    | Channel with cylinder            | Flat heated plate, no cylinder  |
| Fields      | Single velocity `v`              | `(ux, uy, T, pf)`               |
| Data format | `.npy`, shape `[N,X,Y,T]`        | HDF5 (`.h5`)                    |
| Boundary    | Cylinder no-slip + channel walls | Bottom heated wall + side walls |
| Phenomenon  | Vortex shedding                  | Nucleate/transitional boiling   |

These differences mean that while DiffusionPDE provides a solid diffusion-model framework to build from, several components — the data loader, the boundary mask, the PDE residual, and the network input channels — will need to be adapted for BubbleML.
