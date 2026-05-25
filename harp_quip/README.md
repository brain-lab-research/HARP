# HARP: Hadamard-Preconditioned Adaptive Rotations for Extreme LLM Quantization (QuIP# backend)

This directory is a fork of the **QuIP#** codebase with added support for **HARP** (Hadamard-Preconditioned Adaptive Rotations), a learnable structured orthogonal incoherence processor fit from calibration statistics. In this backend, HARP is used as a drop-in replacement for QuIP#'s fixed randomized Hadamard/RHT mixing stage while keeping the rest of the QuIP# pipeline unchanged: codebooks, LDLQ/solver logic, HF export, and evaluation flow.

The main comparison mode used in the paper is **no weight fine-tuning** (`--ft_epochs 0`), so that the only changed component is the incoherence processor (`--incoh_mode had` vs. `--incoh_mode harp`). The code also keeps the original QuIP# fine-tuning path available.

---

## What is new vs. upstream QuIP#

- **HARP modules and fitting code** in `lib/incoherence/`:
  - mixed-radix stride stages,
  - optional Kronecker fallback,
  - Hadamard/QR fixed mixers,
  - Cayley/Givens orthogonal parameterization,
  - diagonal Hessian-weighted proxy loss,
  - optional block-diagonal Hessian regularization.
- **QuIP# integration** via `--incoh_mode harp`; use `--incoh_mode had` for the original fixed Hadamard/RHT baseline and `--incoh_mode kron` for the original Kronecker-style mode.
- **2/3/4-bit codebooks**:
  - `E8P12` for 2-bit,
  - `E8P12RVQ3B` for 3-bit,
  - `E8P12RVQ4B` for 4-bit.
- **HARP parameter storage accounting and optional int8 export** through `quantize_llama/hfize_llama.py --quantize_harp_theta`.
- **Calibration-time controls**, including target refresh with `--harp_q_recompute_every` and layer timing summaries.
- **Large-run utilities**, including single-layer quantization, local resume, optional Hugging Face layer-artifact upload/resume, and multi-GPU dynamic scheduling in the no-finetuning path.
- **Paper reproduction scripts** in `harp_scripts/` for Llama 2, Llama 3.2, 2/3/4-bit settings, perplexity, zero-shot evaluation, and speed evaluation.

---

## Repository layout

- `lib/incoherence/` **(main HARP addition)**
  - `harp.py`: two-sided HARP processor, fitting loop, proxy objective, storage accounting.
  - `harp_lib.py`: mixed-radix staged orthogonal transform implementation and optional Kronecker fallback.
  - `loss_utils.py`: chunked codebook quantization used for the HARP calibration proxy.
  - `base.py`: incoherence processor interface.

- `lib/codebook/`
  - QuIP# E8P12 codebook plus RVQ 3-bit and 4-bit variants.
  - HARP-aware inference paths for applying the learned pre/post transforms.

- `quantize_llama/`
  - `hessian_offline_llama.py`: generate Hessian/second-moment statistics.
  - `quantize_finetune_llama.py`: layerwise quantization and optional HARP fitting.
  - `hfize_llama.py`: export quantized checkpoints to Hugging Face format; supports `--quantize_harp_theta`.
  - `finetune_e2e_llama.py`: original-style end-to-end fine-tuning utility.

- `harp_scripts/`
  - One-shot scripts for paper-style quantization, export, and evaluation.

- `eval/`
  - `eval_ppl.py`: WikiText2/C4 perplexity.
  - `eval_zeroshot.py`: lm-eval zero-shot tasks.
  - `eval_speed.py`: decoding-style speed benchmark.
  - `interactive_gen.py`: interactive generation utility.

- `quiptools/`
  - QuIP# CUDA kernels and build scripts.

Most other files follow upstream QuIP#, with small changes to pass HARP configuration/state through quantization, HF export, and inference.

---

## Installation

Follow the QuIP# installation flow.

1. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Build and install QuIP# CUDA kernels**

   ```bash
   cd quiptools
   python setup.py install
   cd ..
   ```

3. **Install fast-hadamard-transform**

   Install from source if possible, or via pip if it works in your environment:

   ```bash
   pip install fast-hadamard-transform
   ```

   Source repository: `https://github.com/Dao-AILab/fast-hadamard-transform`

4. **Set model/data access tokens if needed**

   The reproduction scripts assume access to gated Llama checkpoints and, when downloading precomputed Hessians or uploading layer artifacts, Hugging Face access tokens:

   ```bash
   export HF_TOKEN=...
   export HF_USER_ACCESS_TOKEN=...
   export HF_UPLOAD_TOKEN=...   # only needed for upload/resume runs
   ```

Notes:

- Use a CUDA/PyTorch stack compatible with your GPUs.
- Hessian generation and HARP fitting can be memory intensive.
- For Llama 2 70B or other long runs, use the resume/upload utilities described below.

---

## Quick start: paper-style HARP run

The easiest way to reproduce a paper-style run is to use a script from `harp_scripts/`. For example:

```bash
bash harp_scripts/quantize_llama2_7b.sh
```

Each script performs the same high-level pipeline:

1. download or reuse Hessian/second-moment statistics,
2. quantize with HARP (`--incoh_mode harp`),
3. export the quantized checkpoint to HF format,
4. evaluate perplexity and zero-shot accuracy.

To run the fixed RHT baseline, set:

```bash
--incoh_mode had
```

To use the optional Kronecker fallback variant for HARP, add:

```bash
--harp_kron_fallback
```

---

## Manual pipeline

### 1. Hessian / second-moment statistics

You can either generate Hessians locally or download precomputed ones used by the scripts.

Generate locally:

```bash
python -m quantize_llama.hessian_offline_llama \
  --base_model meta-llama/Llama-2-7b-hf \
  --save_path hess/llama2_7b_6144 \
  --devset_size 256 \
  --ctx_size 4096
```

Download precomputed statistics:

```bash
python scripts/download_hf.py \
  --folder_path hess/llama2_7b_6144 \
  --repo_id relaxml/Hessians-Llama-2-7b-6144
```

Hessians can be computed once per base model and reused across RHT/HARP, bitwidths, and most HARP hyperparameter sweeps.

### 2. Quantize with HARP

Example 2-bit HARP quantization for Llama 2 7B:

```bash
python -m quantize_llama.quantize_finetune_llama \
  --save_path ckpt/llama2_7b_2bit_harp \
  --ft_train_mode \
  --ft_epochs 0 \
  --codebook E8P12 \
  --scale_override 0.9 \
  --base_model meta-llama/Llama-2-7b-hf \
  --hessian_path hess/llama2_7b_6144 \
  --devset_size 384 \
  --ft_valid_size 128 \
  --incoh_mode harp \
  --harp_b 8 \
  --harp_max_b 8 \
  --harp_passes 1 \
  --harp_ordering_mode stride \
  --harp_fixed_mixer had_or_qr \
  --harp_steps 1200 \
  --harp_strategy proxy \
  --harp_lr_u 0.03 \
  --harp_lr_v 0.03 \
  --harp_hbd_lambda 0.1 \
  --harp_hbd_block 8 \
  --harp_chunk_size 1048576 \
  --harp_q_recompute_every 1
```

For 3-bit and 4-bit, change the codebook:

```bash
--codebook E8P12RVQ3B   # 3-bit
--codebook E8P12RVQ4B   # 4-bit
```

The paper-style 3/4-bit scripts also use:

```bash
--scale_override -1 --resid_scale_override -1
```

### 3. Export to Hugging Face format

```bash
python -m quantize_llama.hfize_llama \
  --quantized_path ckpt/llama2_7b_2bit_harp \
  --hf_output_path hf/llama2_7b_2bit_harp
```

To export the compact int8 HARP-parameter variant and report BPP with int8 HARP storage, add:

```bash
--quantize_harp_theta
```

Example:

```bash
python -m quantize_llama.hfize_llama \
  --quantized_path ckpt/llama2_7b_2bit_harp \
  --hf_output_path hf/llama2_7b_2bit_harp_int8theta \
  --quantize_harp_theta
```

### 4. Evaluate

Perplexity:

```bash
python -m eval.eval_ppl \
  --hf_path hf/llama2_7b_2bit_harp \
  --seqlen 4096
```

For Llama 3.2 experiments, use the longer context length used in the paper:

```bash
python -m eval.eval_ppl \
  --hf_path hf/llama32_1b_2bit_harp \
  --seqlen 8192
```

Zero-shot evaluation:

```bash
python -m eval.eval_zeroshot \
  --hf_path hf/llama2_7b_2bit_harp \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 4
```

Speed benchmark:

```bash
python -m eval.eval_speed \
  --hf_path hf/llama2_7b_2bit_harp \
  --batch_size 1 \
  --seqlen 1 \
  --samples 2000
```

Use `--no_use_cuda_graph` or `--no_use_flash_attn` if your environment does not support those paths.

---

## HARP flags

### Core mode flags

- `--incoh_mode harp`: enable HARP.
- `--incoh_mode had`: original QuIP# fixed Hadamard/RHT baseline.
- `--incoh_mode kron`: original Kronecker-style mode.
- `--harp_kron_fallback`: use HARP's optional Kronecker fallback when compatible.

### Structure / initialization

- `--harp_b 8`: preferred radix for the greedy mixed-radix schedule.
- `--harp_max_b 8`: maximum radix used when factoring the remaining dimension.
- `--harp_passes 1`: number of independent HARP passes.
- `--harp_ordering_mode stride`: stride-stage execution order used in the paper.
- `--harp_fixed_mixer had_or_qr`: Hadamard for power-of-two radices; deterministic QR fallback otherwise.
- `--harp_use_givens_b2`: use a Givens angle parameterization for radix-2 stages.
- `--harp_theta_init_scale 0.0`: initialize learnable rotations to the Hadamard/RHT-equivalent point.
- `--harp_theta_clip`: optional clipping for skew parameters.

### Fitting objective

- `--harp_steps 1200`: HARP optimizer steps per layer/module.
- `--harp_strategy proxy`: diagonal Hessian-weighted proxy objective.
- `--harp_lr_u`, `--harp_lr_v`: learning rates for output- and input-side rotations.
- `--harp_hbd_lambda 0.1`: block-diagonal Hessian regularization weight.
- `--harp_hbd_block 8`: block size for the Hessian block-diagonalization regularizer.
- `--harp_reg_theta`: optional L2 regularization on HARP parameters.
- `--harp_grad_clip`: optional gradient clipping.

### Calibration speed / memory

- `--harp_chunk_size`: chunk size for calibration-time codebook quantization. Larger values are faster but use more VRAM.
- `--harp_q_recompute_every K`: recompute the stopped-gradient quantized target every `K` optimizer steps. `K=1` is the default and matches the original behavior; larger values can reduce calibration time.

---

## Layer timing, resume, and long-run utilities

The updated quantization script writes one timing file per layer:

```text
<save_path>/<layer_idx>_timing.json
```

At the end of a run it logs:

- average calibration time per module type (`qkv`, `o`, `up`, `down`),
- average calibration time per layer block,
- summed layer-block calibration time,
- total wall-clock time.

For debugging or partial quantization:

```bash
--layer_idx 12
```

quantizes only one decoder layer.

For local resume / slicing long runs:

```bash
--resume_from_layer 10
--stop_before_layer 20
```

In the no-finetuning path (`--ft_epochs 0`), the script uses dynamic multi-GPU layer scheduling when multiple CUDA devices are visible.

For very long jobs, the code can upload each completed layer to a Hugging Face repository and skip layers that already exist remotely:

```bash
--upload_after_layer \
--upload_repo_id <namespace/repo> \
--upload_token_env HF_UPLOAD_TOKEN \
--upload_lock_path /tmp/harp_hf_upload.lock
```

To resume from already-uploaded remote artifacts:

```bash
--skip_hf_existing \
--upload_repo_id <namespace/repo> \
--upload_token_env HF_UPLOAD_TOKEN
```

This path is intended for `--ft_epochs 0`. The script will reject remote-skip/resume with fine-tuning enabled because activation replay needs additional care.

---

## Paper scripts

Available scripts include:

- `harp_scripts/quantize_llama2_7b.sh`: Llama 2 7B, 2-bit HARP.
- `harp_scripts/quantize_llama2_13b.sh`: Llama 2 13B, 2-bit HARP.
- `harp_scripts/quantize_llama2_70b.sh`: Llama 2 70B, 2-bit HARP and RHT baseline.
- `harp_scripts/quantize_llama2_7b_13b_3bit.sh`: Llama 2 7B/13B, 3-bit HARP and RHT baselines.
- `harp_scripts/quantize_llama2_7b_13b_4bit.sh`: Llama 2 7B/13B, 4-bit HARP and RHT baselines.
- `harp_scripts/quantize_llama32_1b.sh`: Llama 3.2 1B, 2-bit HARP.
- `harp_scripts/quantize_llama32_3b.sh`: Llama 3.2 3B, 2-bit HARP.

The Llama 2 paper tables use context length 4096 for the main QuIP# comparison. The Llama 3.2 scripts use context length 8192.

---

## Notes and troubleshooting

- **Hessian reuse:** Hessian/second-moment files are expensive to generate but can be reused across HARP/RHT and across bitwidths for the same model and calibration setup.
- **HARP storage:** use `hfize_llama.py --quantize_harp_theta` for the int8 HARP-parameter storage variant used in BPP/storage reporting.
- **Memory:** reduce `--harp_chunk_size` if HARP fitting OOMs. Smaller chunks are slower but reduce peak VRAM.
- **Calibration time:** increasing `--harp_q_recompute_every` can substantially reduce fitting cost, but may reduce quality if set too high.
- **CUDA graphs:** speed evaluation is usually more meaningful with CUDA graphs enabled. Use `--no_use_cuda_graph` only if needed for compatibility/debugging.
- **Fine-tuning:** the main paper results use `--ft_epochs 0` to isolate HARP. The original QuIP# fine-tuning path remains available, and HARP parameters are assigned their own fine-tuning learning rate via `--ft_harp_lr` when present.

---

## License

This repository inherits QuIP#'s licensing (**GNU GPL v3**). Model checkpoints and datasets remain governed by their original licenses, including the Meta Llama license where applicable.
