#!/usr/bin/env bash
set -euo pipefail

# export CUDA_HOME="$CONDA_PREFIX"
# export CUDA_PATH="$CONDA_PREFIX"
# export PATH="$CONDA_PREFIX/bin:$PATH"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_USER_ACCESS_TOKEN=<TOKEN>
export HF_TOKEN=<TOKEN>

CKPT=ckpt
HF=hf
LOG=log
HESS=hess

# HARP-QTIP 2 bit
K="2"   # bitwidth
OUT_NAME="qtip_llama2_7b_${K}bit_harp"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_7b"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG" "$HESS_PATH"

hf download meta-llama/Llama-2-7b-hf
python scripts/download_hf.py --folder_path "$HESS_PATH" --repo_id relaxml/Hessians-Llama-2-7b-6144

python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --codebook bitshift \
  --base_model meta-llama/Llama-2-7b-hf \
  --in_hess_path "$HESS_PATH" \
  --scale_override 0.9 \
  --ft_epochs 0 \
  --td_x 16 \
  --td_y 16 \
  --L 16 \
  --K "$K" \
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
  --harp_reg_theta 0.0 \
  --harp_theta_init_scale 0.0 \
  --harp_hbd_lambda 0.1 \
  --harp_hbd_block 16 \
  --harp_chunk_size 1048576 \
  --harp_q_recompute_every 2 \
  --harp_q_recompute_every_final 2 \
  --harp_q_refresh_parts 16 \
  >> "$LOG_PATH" 2>&1

python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

# RHT-QTIP 2 bit
K="2"   # bitwidth
OUT_NAME="qtip_llama2_7b_${K}bit_rht"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_7b"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG" "$HESS_PATH"

python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --codebook bitshift \
  --base_model meta-llama/Llama-2-7b-hf \
  --in_hess_path "$HESS_PATH" \
  --scale_override 0.9 \
  --ft_epochs 0 \
  --td_x 16 \
  --td_y 16 \
  --L 16 \
  --K "$K" \
  --V 2 \
  --decode_mode quantlut_sym \
  --tlut_bits 9 \
  --incoh_mode had \
  >> "$LOG_PATH" 2>&1

python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1
