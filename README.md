# HARP: Hadamard-Preconditioned Adaptive Rotations for Extreme LLM Quantization

This repository contains two backend integrations of **HARP** (Hadamard-Preconditioned Adaptive Rotations), a learnable structured orthogonal incoherence processor fit on calibration data.

HARP is designed to replace a fixed randomized Hadamard mixing stage with a learned orthogonal processor while keeping the downstream quantization backend unchanged. The repository is organized as a two-backend monorepo:

- `harp_quip/`: QuIP# integrated with HARP.
- `harp_qtip/`: QTIP integrated with HARP.

Both integrations share the same high-level idea: set the backend's incoherence processor to `harp` instead of the original fixed Hadamard/RHT processor, fit HARP during calibration, and then export/evaluate the resulting quantized model using the backend's usual flow.

---

## Repository layout

```text
.
├── harp_quip/   # QuIP# + HARP
└── harp_qtip/   # QTIP + HARP
```

Each subdirectory is a self-contained fork of the corresponding upstream codebase. Install and run each backend from inside its own folder.

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

## Licenses and model terms

Each backend folder inherits the license of the corresponding upstream project. Model weights and tokenizers remain governed by their original licenses, for example the Meta Llama license for Llama models.
