#!/usr/bin/env bash
# ==============================================================================
# V-JEPA2  Multi-View  Action Inference  &  Closed-Loop MPC
# ==============================================================================
#
# Multi-camera variant of the action inference script. Evaluates the predictor
# trained with multiple camera views and learnable view embeddings.
#
# Each action in ACTIONS defines a constant-action GT trajectory. For each one
# the script:
#   1. Rolls out the GT trajectory in the env.
#   2. Visualises the energy landscape (predictor loss over sampled actions).
#   3. Runs closed-loop MPC toward the goal (last GT frame representation).
#   4. Produces a side-by-side GIF (GT vs MPC) and 3D trajectory plot.
#
# At the end a combined summary GIF, summary trajectory plot, and summary JSON
# are generated across all actions.
#
# Usage:
#   chmod +x run_action_inference_multiview.sh
#   ./run_action_inference_multiview.sh
#
# ==============================================================================
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# TASK
# ──────────────────────────────────────────────────────────────────────
TASK="pick-place-v3"
EPISODE_LENGTH=40
SEED=0

# Observation cameras — must match training order.
CAMERA_NAMES="topview back"

# Camera used only for rendered GIF output.
RENDER_CAMERA="corner"

IMAGE_SIZE=224

# Extra env kwargs (space-separated key=value pairs).
ENV_KWARGS=""

# ──────────────────────────────────────────────────────────────────────
# ACTIONS
# ──────────────────────────────────────────────────────────────────────
# Each action is [dx, dy, dz, gripper]. Applied as a constant action
# for EPISODE_LENGTH steps to create the GT trajectory. The MPC agent
# then tries to reproduce it.
#
# Format: JSON array of arrays.
#   Single action:    '[[0.0, 0.0, 0.2, 0.0]]'
#   Multiple actions: '[[0.0, 0.0, 0.2, 0.0], [0.02, 0.03, -0.01, 0.3], [-0.1, 0.0, 0.0, 0.0]]'
ACTIONS='[
    [0.0,  0.0,  -0.2, 0.0],
    [0.2, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.0, 0.0]
]'

# ──────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────
MODEL="large"
ENCODER_CKPT="/Metaworld/third_party/vjepa2/ckpts/vitl.pt"
PREDICTOR_CKPT="/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_multiview_0403/latest.pt"

# ──────────────────────────────────────────────────────────────────────
# ENERGY LANDSCAPE
# ──────────────────────────────────────────────────────────────────────
ENERGY_NSAMPLES=5
ENERGY_GRID_SIZE=0.01
ENERGY_ACTION_REPEAT=1

# ──────────────────────────────────────────────────────────────────────
# MPC / CEM
# ──────────────────────────────────────────────────────────────────────
MPC_ROLLOUT=2
MPC_SAMPLES=500
MPC_TOPK=10
MPC_CEM_STEPS=15
MPC_MOMENTUM_MEAN=0.15
MPC_MOMENTUM_MEAN_GRIPPER=0.15
MPC_MOMENTUM_STD=0.75
MPC_MOMENTUM_STD_GRIPPER=0.15
MPC_MAXNORM=0.1
MAX_MPC_STEPS=100
GOAL_THRESHOLD=0.01

# ──────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="./output_inference_multiview/${TASK}_${MODEL}_seed${SEED}"
GIF_FPS=10

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

# Collapse ACTIONS to a single line for the CLI
ACTIONS_ONELINE=$(echo "$ACTIONS" | tr -d '\n' | tr -s ' ')

CMD=(
    python action_inference_multiview.py
    --task              "$TASK"
    --episode-length    "$EPISODE_LENGTH"
    --seed              "$SEED"
    --camera-names      $CAMERA_NAMES
    --render-camera     "$RENDER_CAMERA"
    --image-size        "$IMAGE_SIZE"
    --model             "$MODEL"
    --actions           "$ACTIONS_ONELINE"
    --energy-nsamples   "$ENERGY_NSAMPLES"
    --energy-grid-size  "$ENERGY_GRID_SIZE"
    --energy-action-repeat "$ENERGY_ACTION_REPEAT"
    --mpc-rollout       "$MPC_ROLLOUT"
    --mpc-samples       "$MPC_SAMPLES"
    --mpc-topk          "$MPC_TOPK"
    --mpc-cem-steps     "$MPC_CEM_STEPS"
    --mpc-momentum-mean          "$MPC_MOMENTUM_MEAN"
    --mpc-momentum-mean-gripper  "$MPC_MOMENTUM_MEAN_GRIPPER"
    --mpc-momentum-std           "$MPC_MOMENTUM_STD"
    --mpc-momentum-std-gripper   "$MPC_MOMENTUM_STD_GRIPPER"
    --mpc-maxnorm       "$MPC_MAXNORM"
    --max-mpc-steps     "$MAX_MPC_STEPS"
    --goal-threshold    "$GOAL_THRESHOLD"
    --output-dir        "$OUTPUT_DIR"
    --gif-fps           "$GIF_FPS"
)

# Add env kwargs if set
if [[ -n "${ENV_KWARGS:-}" ]]; then
    CMD+=(--env-kwargs $ENV_KWARGS)
fi

# Add checkpoint overrides
if [[ -n "${ENCODER_CKPT:-}" ]]; then
    CMD+=(--encoder-ckpt "$ENCODER_CKPT")
fi
if [[ -n "${PREDICTOR_CKPT:-}" ]]; then
    CMD+=(--predictor-ckpt "$PREDICTOR_CKPT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
