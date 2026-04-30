#!/usr/bin/env bash
# ==============================================================================
# V-JEPA2  MPC  Evaluation  —  Expert-Guided Task Completion
# ==============================================================================
#
# This script runs a closed-loop MPC evaluation where the V-JEPA2 world model
# tries to replicate an expert demonstration by tracking intermediate goal
# frames sampled from the expert rollout.
#
# Pipeline:
#   1. Two identical Metaworld environments are created (expert + MPC).
#   2. The expert policy rolls out, and K intermediate goal frames are sampled.
#   3. Each goal frame is encoded into the V-JEPA2 latent space.
#   4. The MPC controller (CEM) plans actions to move the current latent
#      representation toward each goal representation in sequence.
#   5. Goal switching happens when the representation distance drops below a
#      threshold, or the per-goal step budget is exhausted.
#   6. A side-by-side GIF, 3D trajectory plot, and analysis logs are saved.
#
# Usage:
#   chmod +x run_mpc_eval.sh
#   ./run_mpc_eval.sh
#
# ==============================================================================
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# TASK CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
# Task name. Supported: pick-place-v3, drawer-open-v3, door-open-v3,
#   door-close-v3, door-unlock-v3, door-lock-v3, assembly-v3, disassemble-v3
TASK="pick-place-v3"

# How many env steps the expert policy executes.
EPISODE_LENGTH=200

# Random seed — both envs are created with this seed so they start identically.
SEED=42

# Camera for observations AND rendered GIF frames.
CAMERA_NAME="corner"

# Observation image size (square). Must match the model's expected input.
IMAGE_SIZE=224

# Extra env constructor kwargs (space-separated key=value pairs).
# For pick-place-v3, use initialise_region=fixed to make both envs identical.
# For other tasks, the seed-based task system already provides reproducibility.
ENV_KWARGS="initialise_region=fixed"

# ──────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
# Encoder backbone: "giant" (ViT-G, ~1B params) or "large" (ViT-L, ~300M).
MODEL="large"

# Override checkpoint paths (leave empty to use defaults for the chosen model).
ENCODER_CKPT="/Metaworld/third_party/vjepa2/ckpts/vitl.pt"
PREDICTOR_CKPT="/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_corner_0304/latest.pt"

# ──────────────────────────────────────────────────────────────────────
# INTERMEDIATE GOAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
# Number of intermediate goal frames evenly sampled from the expert rollout.
# E.g. 5 goals from a 200-step rollout → goals at steps 40, 80, 120, 160, 200.
# More goals = finer guidance but more switching overhead.
N_GOALS=5

# ──────────────────────────────────────────────────────────────────────
# MPC / CEM PARAMETERS
# ──────────────────────────────────────────────────────────────────────
# These control the Cross-Entropy Method (CEM) optimisation that the MPC
# uses to select actions at each step.
#
# Source: V-JEPA 2 paper (arxiv:2506.09985)
#   Section 4, §4.1, §4.2, §11.2, Table 3
#
# Paper configuration for V-JEPA 2-AC on real Franka:
#   - 800 samples, 10 refinement steps, top-10 elites, horizon=1
#   - Actions constrained to L1-ball of radius 0.075 (~13cm max EE displacement)
#   - Gaussian init: N(0, 1) mean, unit variance
#   - 16 seconds per action on single RTX 4090
#
# --- Planning horizon ---
# Paper §11.2: "Since all considered tasks are relatively greedy, we found
#   a short planning horizon to be sufficient for our setup."
MPC_ROLLOUT=2

# --- Sampling ---
# Paper §11.2/Table 3: "we use 800 samples, 10 refinement steps based on
#   the top 10 samples from the previous iteration"
MPC_SAMPLES=800

# Paper §11.2: "top 10 samples from the previous iteration"
MPC_TOPK=10

# --- Optimisation ---
# Paper §11.2/Table 3: "10 refinement steps"
MPC_CEM_STEPS=10

# --- Momentum (distribution update smoothing) ---
# The paper describes standard CEM (full replacement of distribution with
# elite statistics each iteration, i.e. momentum=0). We add a small amount
# of momentum for stability in simulation.
# 0.0 = standard CEM (paper).
# Small positive values add smoothing (recommended for sim).
MPC_MOMENTUM_MEAN=0.1
MPC_MOMENTUM_MEAN_GRIPPER=0.1
MPC_MOMENTUM_STD=0.5
MPC_MOMENTUM_STD_GRIPPER=0.1

# --- Action magnitude ---
# Paper §4.1: "we constrain each sampled action to the L1-Ball of radius
#   0.075 centered at the origin, which corresponds to a maximum
#   end-effector displacement of approximately 13 cm for each individual
#   action, since large actions are relatively out-of-distribution."
# NOTE: 0.075 is for Droid (real Franka).  For MetaWorld, the env applies
#   action_scale = 1/80 inside set_xyz_action.  Setting maxnorm = 0.0125
#   ensures CEM-planned actions (in meters) convert to MW actions within
#   [-1, 1] without clipping:  mw_action = cem_xyz / (1/80) = cem_xyz * 80.
#   With maxnorm=0.0125:  0.0125 * 80 = 1.0  (exactly full range).
MPC_MAXNORM=0.0125

# ──────────────────────────────────────────────────────────────────────
# GOAL SWITCHING CRITERIA
# ──────────────────────────────────────────────────────────────────────
# The MPC moves to the next intermediate goal when EITHER condition is met:
#
# Paper §4.2/§11.2: The paper uses FIXED step counts for sub-goal switching
#   on real Franka pick-and-place (3 sub-goals with 4+10+4 = 18 total steps).
#   We use representation-distance based switching (as requested) with a hard
#   step budget fallback.
#
# GOAL_REP_THRESHOLD: Representation-space L1 distance between the
#   current encoded observation and the goal encoding. When the distance
#   drops below this value, we consider the goal "reached" and switch.
#   Lower  → the robot must match the goal very precisely before moving on.
#   Higher → quicker switching, tolerant of approximate matching.
#   The right value depends on the encoder; start with 0.3 and tune.
GOAL_REP_THRESHOLD=0.3

# MAX_STEPS_PER_GOAL: Hard budget per goal. Paper uses 4-10 steps per
#   sub-goal on real Franka at 4fps. In simulation, more steps may be needed.
#   If the rep threshold is never reached, we move on after this many steps.
MAX_STEPS_PER_GOAL=25

# MAX_TOTAL_STEPS: Overall MPC step budget across all goals.
#   Paper's pick-and-place: 18 total steps (3 sub-goals).
#   We allow more since sim dynamics/action scaling differ from real robot.
MAX_TOTAL_STEPS=200

# ──────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR="./output_mpc_eval_031/${TASK}_${MODEL}_seed${SEED}"
GIF_FPS=15

# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

CMD=(
    python mpc_eval.py
    --task              "$TASK"
    --episode-length    "$EPISODE_LENGTH"
    --seed              "$SEED"
    --camera-name       "$CAMERA_NAME"
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
