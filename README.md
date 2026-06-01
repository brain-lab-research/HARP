# HARP: Hadamard-Preconditioned Adaptive Rotations for Extreme LLM Quantization

<p align="center">
  <a href="https://arxiv.org/abs/2605.29843">
    <img src="https://img.shields.io/badge/arXiv-2605.29843-b31b1b.svg">
  </a>
</p>

<p align="center">
  <b>Learnable orthogonal preprocessing for extreme LLM quantization</b>
</p>

> **TL;DR**
>
> HARP replaces the fixed randomized Hadamard transform used in SOTA PTQ pipelines with a learnable structured orthogonal processor trained only on calibration data. It improves quantization quality while preserving the original quantizer, deployment pipeline, and inference efficiency.

## What is HARP?

Modern ultra-low-bit quantization methods such as QuIP# rely on randomized orthogonal transforms (typically Randomized Hadamard Transforms, RHT) to improve weight incoherence before quantization.

HARP asks a simple question:

> **Can the incoherence processor itself be learned instead of fixed?**

To answer this, **HARP** (Hadamard-Preconditioned Adaptive Rotations) introduces a structured orthogonal processor that:

* starts from a Hadamard-style initialization,
* remains computationally efficient,
* is trained only during calibration,
* requires **no model retraining**,
* can be integrated into existing PTQ pipelines with minimal changes.

The resulting processor better adapts to the statistics of each layer, reducing quantization error while preserving the original backend workflow.

---

## Repository layout

The repository is organized as a two-backend monorepo:

```text
.
├── harp_quip/   # QuIP# + HARP
└── harp_qtip/   # QTIP + HARP
```

Both integrations share the same high-level idea: set the backend's incoherence processor to `harp` instead of the original fixed Hadamard/RHT processor, fit HARP during calibration, and then export/evaluate the resulting quantized model using the backend's usual flow. Each subdirectory is a self-contained fork of the corresponding upstream codebase. Install and run each backend from inside its own folder.

---

## Which backend should I use?

Use `harp_quip/` for the main QuIP# experiments, including the paper's primary RHT-vs-HARP comparisons.

Use `harp_qtip/` for the QTIP integration / portability experiment. The HARP module is conceptually the same, but QTIP uses a different trellis-coded quantization backend and its own kernels.

---

## Quick start

### QuIP# + HARP

```bash
cd harp_quip
pip install -r requirements.txt
cd quiptools
python setup.py install
cd ..
```

Then see `harp_quip/README.md` and the scripts under `harp_quip/harp_scripts/`.

### QTIP + HARP

```bash
cd harp_qtip
pip install -r requirements.txt
cd qtip-kernels
python setup.py install
cd ..
```

Then see `harp_qtip/README.md` and the scripts under `harp_qtip/harp_scripts/`.

---

## Main HARP flags

Both backend integrations expose HARP through the incoherence-processing flag:

- `--incoh_mode harp`: use HARP.
- `--incoh_mode had`: use the original fixed Hadamard/RHT processor.

Common HARP options include:

- `--harp_steps`: number of HARP fitting steps per layer/module.
- `--harp_b`, `--harp_max_b`: preferred and maximum radix for mixed-radix stages.
- `--harp_passes`: number of HARP passes.
- `--harp_ordering_mode stride`: stride-order staged execution.
- `--harp_fixed_mixer had_or_qr`: Hadamard for power-of-two blocks, deterministic QR fallback otherwise.
- `--harp_lr_u`, `--harp_lr_v`: learning rates for the two-sided HARP processors.
- `--harp_hbd_lambda`, `--harp_hbd_block`: Hessian block-diagonalization regularizer settings.

Backend-specific flags and scripts are documented in the corresponding subdirectory README.

---

## Notes on calibration and Hessian statistics

HARP is fit during calibration. This adds a one-time quantization cost but does not require retraining the full model. Hessian / second-moment statistics can be generated once per base model and reused across RHT and HARP runs, as well as across hyperparameter sweeps.

Calibration cost differs by backend. QuIP# + HARP is the primary integration used for the main experiments. QTIP + HARP is currently a portability integration and can be slower because QTIP's trellis-coded quantization target is more expensive to refresh during HARP fitting.

---

## Citation

If you use HARP in your research, please cite:

```bibtex

@article{zagitov2026harp,
  title={HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization}, 
  author={Artur Zagitov and Gleb Molodtsov and Aleksandr Beznosikov},
  year={2026},
  eprint={2605.29843},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.29843}, 
}
```
---

## Licenses and model terms

Each backend folder inherits the license of the corresponding upstream project. Model weights and tokenizers remain governed by their original licenses, for example the Meta Llama license for Llama models.
