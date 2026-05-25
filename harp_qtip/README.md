# HARP: Hadamard-Preconditioned Adaptive Rotations for Extreme LLM Quantization (QTIP + HARP)

This repository is a fork of the **QTIP** codebase with added support for **HARP** (Hadamard-preconditioned Adaptive Rotation Processor), a learnable structured orthogonal incoherence processor fit on calibration data.

Conceptually, HARP is a replacement for QTIP's fixed randomized Hadamard / RHT incoherence-processing stage. The QTIP backend remains unchanged: trellis codebooks, quantization parameters, solver flow, HF export, and evaluation scripts follow the QTIP pipeline unless a HARP-specific flag is enabled.

**What is new vs. upstream QTIP**

- **HARP modules and fitting code** in `lib/incoherence/`:
  stride stages, mixed-radix schedules, optional Kronecker fallback, parameterization, losses, and regularizers.
- **QTIP integration** through `--incoh_mode harp` in `quantize_llama/quantize_finetune_llama.py`.
- **HARP-aware QTIP export** with optional int8 HARP-parameter storage through `--quantize_harp_theta` in `quantize_llama/hfize_llama.py`.
- **Target-refresh controls** for reducing HARP calibration cost with QTIP's more expensive quantized target computation.
- **Reproduction scripts** in `harp_scripts/` for running HARP-QTIP and RHT-QTIP comparisons.
- Minor pipeline modifications to save, reload, export, and evaluate HARP processors together with QTIP quantized layers.

---

## Repository layout

- `lib/incoherence/` **(main HARP addition)**
  - `harp.py`: HARP processor wrapper, two-sided transform, and fitting logic.
  - `harp_lib.py`: staged/mixed-radix HARP implementation and orthogonal parameterization.
  - `loss_utils.py`: chunked codebook quantization utilities used during HARP calibration.
  - `base.py`: incoherence processor interface.

- `lib/codebook/`
  - `bitshift.py`: QTIP bitshift/trellis codebook implementation, extended to work with HARP processors during reconstruction and inference.

- `lib/algo/`
  - `finetune.py`: QTIP layer quantization and fine-tuning flow, extended with HARP fitting and save/load support.

- `quantize_llama/`
  - `quantize_finetune_llama.py`: end-to-end Llama quantization entry point.
  - `hfize_llama.py`: export quantized checkpoints to Hugging Face format; supports `--quantize_harp_theta`.
  - `input_hessian_llama.py`: Hessian / second-moment generation.
  - `manifest_model.py`: optional conversion / manifest utility.

- `harp_scripts/` **(main reproduction scripts)**
  - Example scripts for quantizing, exporting, and evaluating QTIP with either fixed RHT or HARP.

- `eval/`
  - Perplexity, zero-shot, and interactive generation utilities.

- `qtip-kernels/`
  - QTIP CUDA kernels and build scripts for fast inference.

Most other folders follow upstream QTIP, with light modifications to support HARP end-to-end.

---

## Installation

Use a CUDA + PyTorch environment compatible with your GPU. The commands below summarize the QTIP installation flow with HARP additions.

1. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Build and install QTIP kernels**

   ```bash
   cd qtip-kernels
   python setup.py install
   cd ..
   ```

3. **Install fast-hadamard-transform**

   Install from source if the pip package does not work in your environment:

   ```bash
   pip install fast-hadamard-transform
   ```

   or build from the upstream source package.

Notes:

- A working CUDA toolkit and PyTorch installation are required.
- Some quantization and HARP fitting runs are memory intensive.
- If you use gated Llama models, make sure your Hugging Face token has access to the base model and tokenizer.

---

## Running QTIP with HARP

The main switch is:

- `--incoh_mode harp`: use HARP as the incoherence processor.
- `--incoh_mode had`: use QTIP's original fixed Hadamard/RHT processor.

A minimal HARP-QTIP quantization command has the following structure:

```bash
python -m quantize_llama.quantize_finetune_llama \
  --save_path ckpt/qtip_llama2_7b_2bit_harp \
  --codebook bitshift \
  --base_model meta-llama/Llama-2-7b-hf \
  --in_hess_path hess/llama2_7b \
  --scale_override 0.9 \
  --ft_epochs 0 \
  --td_x 16 \
  --td_y 16 \
  --L 16 \
  --K 2 \
  --V 2 \
  --decode_mode quantlut_sym \
  --tlut_bits 9 \
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
  --harp_hbd_block 16 \
  --harp_chunk_size 1048576 \
  --harp_q_recompute_every 2 \
  --harp_q_recompute_every_final 2 \
  --harp_q_refresh_parts 16
```

To run the corresponding fixed-RHT QTIP baseline, change only:

```bash
--incoh_mode had
```

and remove the HARP-specific flags.

---

## Exporting to Hugging Face format

After quantization, export the checkpoint:

```bash
python -m quantize_llama.hfize_llama \
  --quantized_path ckpt/qtip_llama2_7b_2bit_harp \
  --hf_output_path hf/qtip_llama2_7b_2bit_harp
```

To store HARP parameters in int8 form during export, add:

```bash
--quantize_harp_theta
```

This affects storage / bit accounting for HARP parameters while keeping the QTIP quantized weights unchanged.

---

## Evaluation

Perplexity evaluation:

```bash
python -m eval.eval_ppl \
  --hf_path hf/qtip_llama2_7b_2bit_harp \
  --seqlen 4096
```

Zero-shot evaluation:

```bash
python -m eval.eval_zeroshot \
  --hf_path hf/qtip_llama2_7b_2bit_harp \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 1
```

For codebooks / decode modes without kernel support, use the manifest options exposed in the evaluation scripts.

---

## Reproduction scripts

The folder `harp_scripts/` contains an example end-to-end script:

```bash
bash harp_scripts/quantize_qtip_harp_llama2_7b.sh
```

The script runs both:

- HARP-QTIP with `--incoh_mode harp`, and
- RHT-QTIP with `--incoh_mode had`.

It downloads or reuses Hessian / second-moment statistics, quantizes the model, exports it to Hugging Face format, and evaluates perplexity.

Before running, edit the script paths and tokens as needed:

- `HF_USER_ACCESS_TOKEN`, `HF_TOKEN`
- `CKPT`, `HF`, `LOG`, `HESS`
- `CUDA_VISIBLE_DEVICES`
- base model path and Hessian path

---

## Important QTIP-specific notes

QTIP exposes the usual trellis / bitshift arguments:

- `--K`: nominal bits per weight.
- `--L`: trellis length.
- `--V`: number of trellis values / branches used by the codebook configuration.
- `--tlut_bits`: tunable lookup-table bits.
- `--decode_mode`: for example `quantlut_sym`, `3inst`, `1mad`, or `lut`.
- `--td_x`, `--td_y`: trellis tile dimensions used in LDLQ.

The current HARP-QTIP integration fits HARP independently for each quantized projection, including the separate query, key, and value projections. This is a straightforward portability integration, but it can be more expensive than the QuIP# integration because QTIP's quantized target computation is heavier. A future optimization is to share the input-side HARP processor across projections with the same input Hessian geometry, while keeping separate output-side processors.

---

## HARP calibration flags

Common HARP flags:

- `--harp_steps`: number of HARP fitting steps per module.
- `--harp_b`, `--harp_max_b`: preferred and maximum mixed-radix stage sizes.
- `--harp_passes`: number of HARP passes.
- `--harp_ordering_mode stride`: stride-order staged execution.
- `--harp_fixed_mixer had_or_qr`: Hadamard for power-of-two blocks, deterministic QR fallback otherwise.
- `--harp_kron_fallback`: optional Kronecker fallback when supported by the dimension.
- `--harp_lr_u`, `--harp_lr_v`: optimizer learning rates for output-side and input-side rotations.
- `--harp_hbd_lambda`, `--harp_hbd_block`: Hessian block-diagonalization regularizer strength and block size.
- `--harp_chunk_size`: chunk size for calibration-time target computation.

Target-refresh flags:

- `--harp_q_recompute_every`: refresh cadence for the stopped-gradient quantized target early in fitting.
- `--harp_q_recompute_every_final`: refresh cadence near the end of fitting.
- `--harp_q_refresh_parts`: number of deterministic interleaved target-cache partitions.

These controls are useful because recomputing the QTIP quantized target is one of the dominant costs in HARP-QTIP calibration.

---

## Troubleshooting

- **Kernel build fails**: verify that CUDA, PyTorch, compiler, and GPU architecture are compatible. Rebuild inside `qtip-kernels/` with `python setup.py install`.
- **Hugging Face access errors**: make sure you have accepted the base model license and exported a valid token.
- **OOM during HARP fitting**: reduce `--harp_chunk_size`, use fewer concurrent GPUs/processes, or quantize a single layer with `--layer_idx` for debugging.
- **Slow calibration**: increase `--harp_q_recompute_every`, increase `--harp_q_recompute_every_final` cautiously, or reduce `--harp_steps` for exploratory runs.
- **Evaluation uses too much memory**: lower `--max_mem_ratio` or use smaller batch sizes in the evaluation scripts.

---

## License

This repository inherits QTIP's license terms. QTIP itself is based on the QuIP# codebase. Model weights and tokenizers remain governed by their original licenses, for example the Meta Llama license for Llama models.
