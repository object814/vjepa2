#!/usr/bin/env bash
# ==============================================================================
# V-JEPA2  Multi-View MPC  Evaluation  —  Expert-Guided Task Completion
# ==============================================================================
#
# Multi-camera variant of run_mpc_eval.sh. Uses a predictor trained with
# multiple camera views and learnable view embeddings.
#
# Key differences from single-view:
#   - Multiple observation cameras (CAMERA_NAMES) are encoded independently.
#   - Per-view learnable embeddings (view_embed) are loaded from the predictor
#     checkpoint and added to encoder outputs before feeding the predictor.
#   - The predictor attention mask is rebuilt for the multi-view token count.
#   - A separate render camera (RENDER_CAMERA) can be used for GIF output.
#
# Usage:
#   chmod +x run_mpc_eval_multiview.sh
#   ./run_mpc_eval_multiview.sh
#
# ==============================================================================
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# TASK CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
TASK="pick-place-v3"
EPISODE_LENGTH=200
SEED=42

# Observation cameras — must match training order exactly.
# These cameras are used for encoding observations into latent space.
CAMERA_NAMES="topview back"

# Render camera — used only for the output GIF (can differ from obs cameras).
RENDER_CAMERA="corner"

IMAGE_SIZE=224

# Extra env constructor kwargs.
ENV_KWARGS="initialise_region=fixed"

# ──────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
MODEL="large"

# Override checkpoint paths (leave empty to use defaults).
ENCODER_CKPT="/Metaworld/third_party/vjepa2/ckpts/vitl.pt"
PREDICTOR_CKPT="/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_multiview_0403/latest.pt"

# ──────────────────────────────────────────────────────────────────────
# INTERMEDIATE GOAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
N_GOALS=5

# ──────────────────────────────────────────────────────────────────────
# MPC / CEM PARAMETERS
# ──────────────────────────────────────────────────────────────────────
MPC_ROLLOUT=2
MPC_SAMPLES=800
MPC_TOPK=10
MPC_CEM_STEPS=10
MPC_MOMENTUM_MEAN=0.1
MPC_MOMENTUM_MEAN_GRIPPER=0.1
MPC_MOMENTUM_STD=0.5
MPC_MOMENTUM_STD_GRIPPER=0.1
MPC_MAXNORM=0.0125

# ──────────────────────────────────────────────────────────────────────
# GOAL SWITCHING CRITERIA
# ──────────────────────────────────────────────────────────────────────
GOAL_REP_THRESHOLD=0.3
MAX_STEPS_PER_GOAL=25
MAX_TOTAL_STEPS=200

# ──────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="./output_mpc_eval_multiview_0403/${TASK}_${MODEL}_seed${SEED}"
GIF_FPS=15

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

CMD=(
    python mpc_eval_multiview.py
    --task              "$TASK"
    --episode-length    "$EPISODE_LENGTH"
    --seed              "$SEED"
    --camera-names      $CAMERA_NAMES
    --render-camera     "$RENDER_CAMERA"
    --image-size        "$IMAGE_SIZE"
    --model             "$MODEL"
    --n-goals           "$N_GOALS"
    --mpc-rollout       "$MPC_ROLLOUT"
    --mpc-samples       "$MPC_SAMPLES"
    --mpc-topk          "$MPC_TOPK"
    --mpc-cem-steps     "$MPC_CEM_STEPS"
    --mpc-momentum-mean          "$MPC_MOMENTUM_MEAN"
    --mpc-momentum-mean-gripper  "$MPC_MOMENTUM_MEAN_GRIPPER"
    --mpc-momentum-std           "$MPC_MOMENTUM_STD"
    --mpc-momentum-std-gripper   "$MPC_MOMENTUM_STD_GRIPPER"
    --mpc-maxnorm       "$MPC_MAXNORM"
    --goal-rep-threshold "$GOAL_REP_THRESHOLD"
    --max-steps-per-goal "$MAX_STEPS_PER_GOAL"
    --max-total-steps    "$MAX_TOTAL_STEPS"
    --output-dir         "$OUTPUT_DIR"
    --gif-fps            "$GIF_FPS"
)

# Add env kwargs if non-empty
if [[ -n "${ENV_KWARGS:-}" ]]; then
    CMD+=(--env-kwargs $ENV_KWARGS)
fi

# Add checkpoint overrides if set
if [[ -n "${ENCODER_CKPT:-}" ]]; then
    CMD+=(--encoder-ckpt "$ENCODER_CKPT")
fi
if [[ -n "${PREDICTOR_CKPT:-}" ]]; then
    CMD+=(--predictor-ckpt "$PREDICTOR_CKPT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
