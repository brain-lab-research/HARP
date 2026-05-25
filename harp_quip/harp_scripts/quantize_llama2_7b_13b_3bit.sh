#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_USER_ACCESS_TOKEN=<TOKEN>
export HF_TOKEN=<TOKEN>
export HF_DATASETS_TRUST_REMOTE_CODE=1

CKPT=ckpt
HF=hf
LOG=log
HESS=hess

# =======================================================
# ====                 7B 3 bits                   ======
# =======================================================

OUT_NAME="2_7b_3bit_harp"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_7b_6144"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG"

# download hessians (comment if already downloaded)
python scripts/download_hf.py --folder_path "$HESS_PATH" --repo_id relaxml/Hessians-Llama-2-7b-6144

# add --harp_kron_fallback to use Kronecker fallback
# reduce --harp_chunk_size if short on VRAM
python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --ft_train_mode \
  --codebook E8P12RVQ3B \
  --scale_override -1 \
  --resid_scale_override -1 \
  --base_model meta-llama/Llama-2-7b-hf \
  --hessian_path "$HESS_PATH" \
  --devset_size 384 \
  --ft_valid_size 128 \
  --ft_epochs 0 \
  --incoh_mode harp \
  \
  --harp_b 8 \
  --harp_max_b 8 \
  --harp_passes 1 \
  \
  --harp_ordering_mode stride \
  --harp_fixed_mixer had_or_qr \
  \
  --harp_steps 1200 \
  --harp_strategy proxy \
  \
  --harp_lr_u 0.03 \
  --harp_lr_v 0.03 \
  --harp_reg_theta 0.0 \
  --harp_theta_init_scale 0.0 \
  --harp_hbd_lambda 0.1 \
  --harp_hbd_block 8 \
  --harp_chunk_size 1048576 \
  --harp_q_recompute_every 1 \
  >> "$LOG_PATH" 2>&1

# Export to HF
python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: PPL
python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  --no_use_cuda_graph \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: Zero-shot tasks
python -m eval.eval_zeroshot \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 64 \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

rm -rf "$HF_PATH"

OUT_NAME="2_7b_3bit_had"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_7b_6144"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG"

python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --ft_train_mode \
  --codebook E8P12RVQ3B \
  --scale_override -1 \
  --resid_scale_override -1 \
  --base_model meta-llama/Llama-2-7b-hf \
  --hessian_path "$HESS_PATH" \
  --devset_size 384 \
  --ft_valid_size 128 \
  --ft_epochs 0 \
  --incoh_mode had \
  >> "$LOG_PATH" 2>&1

# Export to HF
python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: PPL
python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  --no_use_cuda_graph \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: Zero-shot tasks
python -m eval.eval_zeroshot \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 64 \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

rm -rf "$HF_PATH"

# =======================================================
# ====                 13B 3 bits                  ======
# =======================================================

OUT_NAME="2_13b_3bit_harp"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_13b_6144"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG"

# download hessians (comment if already downloaded)
python scripts/download_hf.py --folder_path "$HESS_PATH" --repo_id relaxml/Hessians-Llama-2-13b-6144

python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --ft_train_mode \
  --codebook E8P12RVQ3B \
  --scale_override -1 \
  --resid_scale_override -1 \
  --base_model meta-llama/Llama-2-13b-hf \
  --hessian_path "$HESS_PATH" \
  --devset_size 384 \
  --ft_valid_size 128 \
  --ft_epochs 0 \
  --incoh_mode harp \
  \
  --harp_b 8 \
  --harp_max_b 8 \
  --harp_passes 1 \
  \
  --harp_ordering_mode stride \
  --harp_fixed_mixer had_or_qr \
  \
  --harp_steps 1200 \
  --harp_strategy proxy \
  \
  --harp_lr_u 0.03 \
  --harp_lr_v 0.03 \
  --harp_reg_theta 0.0 \
  --harp_theta_init_scale 0.0 \
  --harp_hbd_lambda 0.1 \
  --harp_hbd_block 8 \
  --harp_chunk_size 1048576 \
  --harp_q_recompute_every 1 \
  >> "$LOG_PATH" 2>&1

# Export to HF
python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: PPL
python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  --no_use_cuda_graph \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: Zero-shot tasks
python -m eval.eval_zeroshot \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 64 \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

rm -rf "$HF_PATH"

OUT_NAME="2_13b_3bit_had"
SAVE_PATH="$CKPT/$OUT_NAME"
HESS_PATH="$HESS/llama2_13b_6144"
HF_PATH="$HF/$OUT_NAME"
LOG_PATH="$LOG/$OUT_NAME.log"
EVAL_LOG_PATH="$LOG/${OUT_NAME}_eval.log"

mkdir -p "$SAVE_PATH" "$HF_PATH" "$LOG"

python -m quantize_llama.quantize_finetune_llama \
  --save_path "$SAVE_PATH" \
  --ft_train_mode \
  --codebook E8P12RVQ3B \
  --scale_override -1 \
  --resid_scale_override -1 \
  --base_model meta-llama/Llama-2-13b-hf \
  --hessian_path "$HESS_PATH" \
  --devset_size 384 \
  --ft_valid_size 128 \
  --ft_epochs 0 \
  --incoh_mode had \
  >> "$LOG_PATH" 2>&1

# Export to HF
python -m quantize_llama.hfize_llama \
  --quantized_path "$SAVE_PATH" \
  --hf_output_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: PPL
python -m eval.eval_ppl \
  --hf_path "$HF_PATH" \
  --no_use_cuda_graph \
  >> "$EVAL_LOG_PATH" 2>&1

# Eval: Zero-shot tasks
python -m eval.eval_zeroshot \
  --tasks arc_challenge,arc_easy,piqa,winogrande \
  --batch_size 64 \
  --hf_path "$HF_PATH" \
  >> "$EVAL_LOG_PATH" 2>&1

rm -rf "$HF_PATH"