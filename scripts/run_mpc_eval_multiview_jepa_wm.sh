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
# TASK="pick-place-v3"
TASK="grasp-v3"
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
PREDICTOR_CKPT="/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_multiview_jepawms/latest.pt"

# ──────────────────────────────────────────────────────────────────────
# INTERMEDIATE GOAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
N_GOALS=3

# ──────────────────────────────────────────────────────────────────────
# MPC / CEM PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# Tuned per jepa-wms (Terver et al. 2025) findings for MetaWorld:
#   - CEM L2 is best overall (L2 cost now used in Python script)
#   - N=300 samples, K=10 topk, J=15 iters is their MetaWorld config
#   - H=6 rollout horizon with m=3 steps executed (their Table S4.1)
#   - For MetaWorld, Adam L2 was actually best but requires code changes;
#     CEM L2 is competitive and doesn't need WorldModel rewrite.
#   - Reduced momentum for more aggressive CEM updates (less smoothing)
MPC_ROLLOUT=3
MPC_SAMPLES=300
MPC_TOPK=10
MPC_CEM_STEPS=15
MPC_MOMENTUM_MEAN=0.0
MPC_MOMENTUM_MEAN_GRIPPER=0.0
MPC_MOMENTUM_STD=0.0
MPC_MOMENTUM_STD_GRIPPER=0.0
MPC_MAXNORM=0.0125

# ──────────────────────────────────────────────────────────────────────
# GOAL SWITCHING CRITERIA
# ──────────────────────────────────────────────────────────────────────
# Threshold lowered because we switched from L1 to L2 distance
# (L2 of layer-normed features is typically ~0.3-0.7x of L1).
# Increased max_steps_per_goal to give the longer-horizon planner time.
GOAL_REP_THRESHOLD=0.15
MAX_STEPS_PER_GOAL=20
MAX_TOTAL_STEPS=400

# ──────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="./output_mpc_eval_multiview_0411_jepa_wm_long/${TASK}_${MODEL}_seed${SEED}"
GIF_FPS=15

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

CMD=(
    python mpc_eval_multiview_jepa_wm.py
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