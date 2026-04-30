#!/usr/bin/env python3
"""
Multi-View MPC Evaluation Script: Expert-Guided Task Completion with V-JEPA2

This is the multi-camera variant of mpc_eval.py. The predictor was trained with
multiple camera views (e.g. topview, back, gripperPOV) and learnable view
embeddings, so encoding and prediction must handle all views jointly.

Key differences from single-view mpc_eval.py:
  - Multiple cameras are encoded independently by the frozen encoder.
  - Per-view learnable embeddings (view_embed) are loaded from the predictor
    checkpoint and added to the encoder outputs before feeding the predictor.
  - Tokens from all views are interleaved per-timestep before prediction:
        [t0_view0, t0_view1, ..., t1_view0, ...]
  - The predictor attention mask is rebuilt for the multi-view token count.
  - tokens_per_frame = num_views * (image_size // patch_size)^2

Pipeline (identical to single-view except for encoding):
  1. Create two identical Metaworld environments (expert + MPC).
  2. Roll out the expert policy, recording observation frames.
  3. Evenly sample K intermediate goal frames from the expert rollout.
  4. Encode each goal frame (all cameras) into multi-view latent representations.
  5. Run closed-loop MPC: at each step, plan an action with CEM that moves
     the current multi-view latent toward the current goal representation.
  6. Switch to the next goal when the representation L1 distance drops below
     a threshold (or a per-goal step budget is exceeded).
  7. Produce a side-by-side GIF comparing expert and MPC rollouts, plus a
     3D end-effector trajectory plot and analysis logs.
"""

import sys
sys.path.insert(0, "..")

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# MetaWorld action-space scaling
# ---------------------------------------------------------------------------
MW_ACTION_SCALE = 1.0 / 80          # MetaWorld's SawyerXYZEnv.action_scale

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.models.utils.modules import build_action_block_causal_attention_mask
from utils.mpc_utils import compute_new_pose
from utils.world_model_wrapper import WorldModel

# ---- Metaworld ----
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld  # noqa: F401  — triggers env registration
from metaworld.wrappers import ProprioMultiImageObsWrapper

# ---- Scripted expert policies ----
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy
from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy
from metaworld.policies.sawyer_grasp_policy import SawyerGraspV3Policy


# =====================================================================
# Helpers
# =====================================================================

POLICY_MAP = {
    "pick-place-v3":    SawyerPickPlaceV3Policy,
    "grasp-v3":         SawyerGraspV3Policy,
    "drawer-open-v3":   SawyerDrawerOpenV3Policy,
    "door-open-v3":     SawyerDoorOpenV3Policy,
    "door-close-v3":    SawyerDoorCloseV3Policy,
    "door-unlock-v3":   SawyerDoorUnlockV3Policy,
    "door-lock-v3":     SawyerDoorLockV3Policy,
    "assembly-v3":      SawyerAssemblyV3Policy,
    "disassemble-v3":   SawyerDisassembleV3Policy,
}


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _find_state_dict(checkpoint, preferred_keys):
    for key in preferred_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key], key
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"], "state_dict"
    if isinstance(checkpoint, dict) and checkpoint:
        first_value = next(iter(checkpoint.values()))
        if torch.is_tensor(first_value):
            return checkpoint, "<root>"
    raise KeyError(
        f"No state_dict found. Tried keys: {preferred_keys} + ['state_dict', '<root>']. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def _load_module_from_ckpt(module, ckpt_path, preferred_keys, module_name):
    checkpoint = robust_checkpoint_loader(ckpt_path, map_location=torch.device("cpu"))
    state_dict, loaded_key = _find_state_dict(checkpoint, preferred_keys)
    state_dict = _sanitize_state_dict(state_dict)
    msg = module.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {module_name} from {ckpt_path} (key='{loaded_key}') → {msg}")


def add_label(frame, text, position="top"):
    """Burn a text label into the top (or bottom) of an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = 4 if position == "top" else img.height - th - 6
    draw.rectangle([x - 2, y - 2, x + tw + 2, y + th + 2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(img)


# =====================================================================
# Environment creation
# =====================================================================

def make_env(task_name, seed, camera_names, image_size, max_episode_steps=500,
             env_kwargs=None):
    """Create a Metaworld env wrapped with multi-camera image observations."""
    extra = dict(env_kwargs) if env_kwargs else {}
    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        seed=seed,
        **extra,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=image_size,
        image_width=image_size,
        camera_names=camera_names,
    )
    return env


# =====================================================================
# Expert rollout
# =====================================================================

def rollout_expert(env, task_name, episode_length, render_camera_name, warmup_steps=5):
    """
    Run the scripted expert policy, returning recorded data.

    Returns
    -------
    frames        : list[np.ndarray]  – observation images  (H, W, 3*N_cam)
    render_frames : list[np.ndarray]  – rendered RGB frames from `render_camera_name`
    proprios      : list[np.ndarray]  – proprioception vectors (7,)
    rewards       : list[float]
    """
    policy_cls = POLICY_MAP.get(task_name)
    if policy_cls is None:
        raise NotImplementedError(
            f"No scripted policy for '{task_name}'. "
            f"Available: {list(POLICY_MAP.keys())}"
        )
    policy = policy_cls()

    obs, _ = env.reset()
    # Warm-up: let the sim settle
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    frames, render_frames, proprios, rewards = [], [], [], []
    render_frames.append(env.render(camera_name=render_camera_name).copy())

    for t in range(episode_length):
        action = policy.get_action(obs["original_obs"])
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["image"])
        render_frames.append(env.render(camera_name=render_camera_name).copy())
        proprios.append(obs["proprio"].copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    print(f"  Expert rollout: {len(frames)} steps, final reward={rewards[-1]:.3f}")
    return frames, render_frames, proprios, rewards


# =====================================================================
# Multi-view encoding helpers
# =====================================================================

def make_multiview_encoder(encoder, transform, camera_names, view_embed,
                           normalize_reps, device):
    """
    Return a function that encodes a multi-camera observation image into
    multi-view latent representations.

    The observation image is (H, W, 3*N_cam) with cameras channel-concatenated
    in the same order as `camera_names`.

    Two modes:
      - for_predictor=True:  add view_embed before interleaving (predictor input)
      - for_predictor=False: no view_embed (goal / comparison target)
    """
    num_views = len(camera_names)

    def encode_single_view(image_np):
        """Encode a single (H, W, 3) image → (1, n_patches, D)."""
        clip = np.expand_dims(image_np, axis=0)   # (1, H, W, 3)
        clip = transform(clip)[None, :]             # (1, C, 1, H, W)
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(device, non_blocking=True)
        h = encoder(clip)
        h = h.view(B, T, -1, h.size(-1))  # (1, 1, n_patches, D)
        return h  # (1, 1, n_patches, D)

    def encode_multiview(image_np, for_predictor=False):
        """
        Encode a multi-camera observation.

        Args:
            image_np: (H, W, 3*N_cam) uint8 array
            for_predictor: if True, add view embeddings (for predictor input)
                           if False, raw encoder output (for goal / loss target)

        Returns:
            h: (1, num_views * n_patches, D) — interleaved multi-view tokens
        """
        h_views = []
        for cam_idx in range(num_views):
            cam_image = image_np[:, :, 3 * cam_idx: 3 * (cam_idx + 1)]
            h_v = encode_single_view(cam_image)  # (1, 1, n_patches, D)
            h_views.append(h_v)

        if for_predictor and view_embed is not None and num_views > 1:
            # Add view embeddings: view_embed is (num_views, 1, D)
            h_views_ve = [
                hv + view_embed[i].unsqueeze(0).unsqueeze(0)  # broadcast over B, T, n_patches
                for i, hv in enumerate(h_views)
            ]
            # Stack views: (1, 1, num_views, n_patches, D)
            h = torch.stack(h_views_ve, dim=2).flatten(2, 3)  # (1, 1, num_views*n_patches, D)
        else:
            h = torch.stack(h_views, dim=2).flatten(2, 3)  # (1, 1, num_views*n_patches, D)

        # Flatten time dim: (1, num_views*n_patches, D)
        h = h.flatten(1, 2)

        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))

        return h

    return encode_multiview


# =====================================================================
# MPC closed-loop controller (multi-view)
# =====================================================================

@torch.no_grad()
def run_mpc(
    env,
    world_model,
    encode_fn,
    tokens_per_frame,
    goal_frames,          # list of K np.ndarray goal images (H, W, 3*N_cam)
    expert_proprios,      # list of proprios at goal timesteps (for EE logging)
    goal_timesteps,       # list of ints – expert timestep for each goal
    render_camera_name,   # camera name for rendering
    max_steps_per_goal,
    max_total_steps,
    goal_rep_threshold,
    warmup_steps,
    device,
):
    """
    Closed-loop MPC using intermediate goal representations (multi-view).

    Returns
    -------
    mpc_render_frames : list[np.ndarray]
    mpc_proprios      : list[np.ndarray]
    log               : dict  (per-step analysis data)
    """

    # --- Encode goal frames (no view embeddings — these are comparison targets) ---
    goal_reps = [encode_fn(gf, for_predictor=False) for gf in goal_frames]
    n_goals = len(goal_reps)
    print(f"  Encoded {n_goals} goal frames into multi-view latent representations.")

    # --- Reset and warm up MPC env ---
    obs, _ = env.reset()
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    # --- Tracking variables ---
    mpc_render_frames = [env.render(camera_name=render_camera_name).copy()]
    mpc_proprios = [obs["proprio"].copy()]
    step_log = []  # per-step analysis

    current_goal_idx = 0
    steps_on_current_goal = 0
    total_steps = 0

    print(f"  Starting MPC loop | {n_goals} goals | "
          f"max {max_steps_per_goal} steps/goal | "
          f"max {max_total_steps} total | "
          f"rep threshold={goal_rep_threshold:.4f}")

    while current_goal_idx < n_goals and total_steps < max_total_steps:
        # Encode current observation (with view embeddings — predictor input)
        z_current = encode_fn(obs["image"], for_predictor=True)
        s_current = (
            torch.from_numpy(obs["proprio"])
            .float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # Goal rep (no view embeddings)
        z_goal = goal_reps[current_goal_idx]

        # For distance comparison, encode current without view embed too
        z_current_raw = encode_fn(obs["image"], for_predictor=False)

        # Compute representation distance to current goal
        # NOTE: L2 distance consistently outperforms L1 for planning cost
        # across all environments (jepa-wms, Terver et al. 2025, Section 5.2).
        diff = z_current_raw[:, :tokens_per_frame] - z_goal[:, :tokens_per_frame]
        rep_dist = torch.sqrt(torch.mean(diff ** 2)).item()

        # Compute EE position distance to expert at the goal timestep
        current_ee = obs["proprio"][:3]
        expert_ee_at_goal = expert_proprios[current_goal_idx][:3]
        ee_dist = float(np.linalg.norm(current_ee - expert_ee_at_goal))

        # Log
        step_log.append({
            "total_step": total_steps,
            "goal_idx": current_goal_idx,
            "goal_timestep": goal_timesteps[current_goal_idx],
            "steps_on_goal": steps_on_current_goal,
            "rep_dist": rep_dist,
            "ee_dist": ee_dist,
            "ee_pos": current_ee.tolist(),
            "expert_ee": expert_ee_at_goal.tolist(),
        })

        # Check if we should switch to next goal
        switch_reason = None
        if rep_dist < goal_rep_threshold:
            switch_reason = "rep_threshold"
        elif steps_on_current_goal >= max_steps_per_goal:
            switch_reason = "max_steps"

        if switch_reason is not None:
            print(f"    Step {total_steps:4d} | "
                  f"Goal {current_goal_idx}/{n_goals-1} → SWITCH ({switch_reason}) | "
                  f"rep_dist={rep_dist:.4f}  ee_dist={ee_dist:.4f}")
            current_goal_idx += 1
            steps_on_current_goal = 0
            if current_goal_idx >= n_goals:
                break
            z_goal = goal_reps[current_goal_idx]

        # Plan action via CEM
        # The world model receives z_current (with view embed) as predictor input
        mpc_action_7d = world_model.infer_next_action(
            z_current, s_current, z_goal
        )
        a7 = mpc_action_7d[0].cpu().numpy()  # (rollout, 7) → take first step
        if a7.ndim == 2:
            a7 = a7[0]

        # --- Action-space conversion (world-model → MetaWorld) ----------
        action_4d = np.zeros(4, dtype=np.float32)
        action_4d[:3] = a7[:3] / MW_ACTION_SCALE   # meters → MW action units
        action_4d[3]  = a7[6]                       # gripper (passthrough)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_4d)
        mpc_render_frames.append(env.render(camera_name=render_camera_name).copy())
        mpc_proprios.append(obs["proprio"].copy())

        total_steps += 1
        steps_on_current_goal += 1

        if total_steps % 10 == 0 or total_steps <= 5:
            print(f"    Step {total_steps:4d} | "
                  f"Goal {current_goal_idx}/{n_goals-1} | "
                  f"rep_dist={rep_dist:.4f}  ee_dist={ee_dist:.4f}  "
                  f"action=({a7[0]:+.3f},{a7[1]:+.3f},{a7[2]:+.3f},{a7[6]:+.3f})")

        if terminated or truncated:
            print(f"    Episode terminated/truncated at step {total_steps}.")
            break

    goals_reached = current_goal_idx
    print(f"  MPC finished: {total_steps} steps, "
          f"{goals_reached}/{n_goals} goals reached/switched.")

    return mpc_render_frames, mpc_proprios, {
        "steps": step_log,
        "total_steps": total_steps,
        "goals_reached": goals_reached,
        "n_goals": n_goals,
    }


# =====================================================================
# Visualisation
# =====================================================================

def make_side_by_side_gif(expert_frames, mpc_frames, output_path, fps=15):
    """Create a GIF with expert (left) and MPC (right) side by side."""
    max_len = max(len(expert_frames), len(mpc_frames))
    while len(expert_frames) < max_len:
        expert_frames.append(expert_frames[-1].copy())
    while len(mpc_frames) < max_len:
        mpc_frames.append(mpc_frames[-1].copy())

    gif_frames = []
    for t in range(max_len):
        left = add_label(expert_frames[t], "Expert")
        right = add_label(mpc_frames[t], "MPC")
        min_h = min(left.shape[0], right.shape[0])
        if left.shape[0] != min_h:
            pil_l = Image.fromarray(left)
            new_w = int(left.shape[1] * min_h / left.shape[0])
            left = np.array(pil_l.resize((new_w, min_h), Image.LANCZOS))
        if right.shape[0] != min_h:
            pil_r = Image.fromarray(right)
            new_w = int(right.shape[1] * min_h / right.shape[0])
            right = np.array(pil_r.resize((new_w, min_h), Image.LANCZOS))
        concat = np.concatenate([left, right], axis=1)
        gif_frames.append(Image.fromarray(concat))

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"  Saved rollout GIF ({len(gif_frames)} frames) → {output_path}")


def make_trajectory_plot(expert_proprios, mpc_proprios, goal_proprios,
                         goal_timesteps, output_path):
    """3D end-effector trajectory comparison plot."""
    expert_ee = np.array([p[:3] for p in expert_proprios])
    mpc_ee = np.array([p[:3] for p in mpc_proprios])
    goal_ee = np.array([p[:3] for p in goal_proprios])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(expert_ee[:, 0], expert_ee[:, 1], expert_ee[:, 2],
            "b-o", label="Expert", markersize=2, linewidth=1.5, alpha=0.7)
    ax.plot(mpc_ee[:, 0], mpc_ee[:, 1], mpc_ee[:, 2],
            "r-s", label="MPC", markersize=2, linewidth=1.5, alpha=0.7)

    for i, (ge, ts) in enumerate(zip(goal_ee, goal_timesteps)):
        ax.scatter(*ge, c="gold", s=120, marker="*", zorder=5, edgecolors="k",
                   label=f"Goal {i} (t={ts})" if i < 3 else "")

    ax.scatter(*expert_ee[0], c="lime", s=100, marker="^", zorder=5,
               edgecolors="k", label="Start")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EE Trajectories: Expert vs MPC (Multi-View)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D trajectory plot → {output_path}")


def make_analysis_plot(log, output_path):
    """Plot representation distance and EE distance over MPC steps."""
    steps = [s["total_step"] for s in log["steps"]]
    rep_dists = [s["rep_dist"] for s in log["steps"]]
    ee_dists = [s["ee_dist"] for s in log["steps"]]
    goal_idxs = [s["goal_idx"] for s in log["steps"]]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.plot(steps, rep_dists, "b-", linewidth=1, label="Rep L2 distance")
    ax1.set_ylabel("Representation Distance (L2)")
    ax1.set_title("MPC Analysis: Distances to Current Goal Over Time (Multi-View)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, ee_dists, "r-", linewidth=1, label="EE L2 distance (m)")
    ax2.set_ylabel("EE Distance (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.step(steps, goal_idxs, "g-", linewidth=2, where="post", label="Current goal index")
    ax3.set_ylabel("Goal Index")
    ax3.set_xlabel("MPC Step")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved analysis plot → {output_path}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="V-JEPA2 Multi-View MPC Evaluation: follow expert demonstrations"
    )
    # --- Task ---
    p.add_argument("--task", type=str, default="pick-place-v3",
                   choices=list(POLICY_MAP.keys()),
                   help="Metaworld task name")
    p.add_argument("--episode-length", type=int, default=200,
                   help="Expert episode length (number of env steps)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible environments")
    p.add_argument("--camera-names", nargs="+", default=["topview", "back", "gripperPOV"],
                   help="Camera names for multi-view observations (order must match training)")
    p.add_argument("--render-camera", type=str, default="front",
                   help="Camera name for rendering output GIFs (can differ from obs cameras)")
    p.add_argument("--image-size", type=int, default=224,
                   help="Observation image size (square, in pixels)")
    p.add_argument("--env-kwargs", nargs="*", default=[],
                   help="Extra env kwargs as key=value (e.g. initialise_region=fixed)")

    # --- Model ---
    p.add_argument("--model", type=str, default="large",
                   choices=["giant", "large"],
                   help="Encoder backbone size")
    p.add_argument("--encoder-ckpt", type=str, default=None,
                   help="Override encoder checkpoint path")
    p.add_argument("--predictor-ckpt", type=str, default=None,
                   help="Override predictor checkpoint path")

    # --- Goals ---
    p.add_argument("--n-goals", type=int, default=5,
                   help="Number of intermediate goal frames to sample from expert rollout; use -1 to use every expert frame")

    # --- MPC / CEM ---
    p.add_argument("--mpc-rollout", type=int, default=1,
                   help="CEM planning horizon (steps ahead to simulate)")
    p.add_argument("--mpc-rollout-parallel", type=int, default=None,
                   help="Max CEM samples to process in parallel on GPU")
    p.add_argument("--mpc-samples", type=int, default=800,
                   help="CEM: action trajectories sampled per iteration")
    p.add_argument("--mpc-topk", type=int, default=10,
                   help="CEM: elite samples to fit the next distribution")
    p.add_argument("--mpc-cem-steps", type=int, default=10,
                   help="CEM: optimization / refinement iterations per action")
    p.add_argument("--mpc-momentum-mean", type=float, default=0.1,
                   help="CEM: momentum for mean update (xyz)")
    p.add_argument("--mpc-momentum-mean-gripper", type=float, default=0.1,
                   help="CEM: momentum for mean update (gripper)")
    p.add_argument("--mpc-momentum-std", type=float, default=0.5,
                   help="CEM: momentum for std update (xyz)")
    p.add_argument("--mpc-momentum-std-gripper", type=float, default=0.1,
                   help="CEM: momentum for std update (gripper)")
    p.add_argument("--mpc-maxnorm", type=float, default=0.0125,
                   help="CEM: per-axis action magnitude clip (meters, world-model space)")

    # --- Goal switching ---
    p.add_argument("--goal-rep-threshold", type=float, default=0.3,
                   help="Representation L1 distance threshold to switch goals")
    p.add_argument("--max-steps-per-goal", type=int, default=25,
                   help="Max MPC steps per goal")
    p.add_argument("--max-total-steps", type=int, default=200,
                   help="Maximum total MPC steps across all goals")

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="./output_mpc_eval_multiview",
                   help="Directory for output files")
    p.add_argument("--gif-fps", type=int, default=15,
                   help="Frames per second for output GIFs")

    return p.parse_args()


def main():
    args = parse_args()

    num_views = len(args.camera_names)

    # --- Model config ---
    MODEL_CONFIGS = {
        "giant": {
            "model_name": "vit_giant_xformers",
            "encoder_ckpt": "/Metaworld/third_party/vjepa2/ckpts/vitg.pt",
            "predictor_ckpt": "/Metaworld/third_party/vjepa2/train/metaworld_predictor_run1/latest.pt",
            "pred_depth": 24,
            "pred_num_heads": 16,
            "pred_embed_dim": 1024,
        },
        "large": {
            "model_name": "vit_large",
            "encoder_ckpt": "/Metaworld/third_party/vjepa2/ckpts/vitl.pt",
            "predictor_ckpt": "/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_multiview/latest.pt",
            "pred_depth": 12,
            "pred_num_heads": 12,
            "pred_embed_dim": 384,
        },
    }
    mcfg = MODEL_CONFIGS[args.model]
    encoder_ckpt = args.encoder_ckpt or mcfg["encoder_ckpt"]
    predictor_ckpt = args.predictor_ckpt or mcfg["predictor_ckpt"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Parse env kwargs ---
    env_kwargs = {}
    for kv in args.env_kwargs:
        if "=" not in kv:
            raise ValueError(f"Invalid env kwarg: {kv}. Expected key=value.")
        k, v = kv.split("=", 1)
        env_kwargs[k] = v

    # --- Compute token counts ---
    single_view_tokens = int((args.image_size // 16) ** 2)  # patch_size=16
    tokens_per_frame = num_views * single_view_tokens

    print("=" * 60)
    print("V-JEPA2  MULTI-VIEW  MPC  EVALUATION")
    print("=" * 60)
    print(f"  Task:            {args.task}")
    print(f"  Model:           {args.model} ({mcfg['model_name']})")
    print(f"  Device:          {DEVICE}")
    print(f"  Camera views:    {args.camera_names} ({num_views} views)")
    print(f"  Render camera:   {args.render_camera}")
    print(f"  Tokens/frame:    {tokens_per_frame} ({num_views} views × {single_view_tokens} patches)")
    print(f"  Expert length:   {args.episode_length}")
    print(f"  N goals:         {args.n_goals}")
    print(f"  Max steps/goal:  {args.max_steps_per_goal}")
    print(f"  Max total steps: {args.max_total_steps}")
    print(f"  Goal threshold:  {args.goal_rep_threshold}")
    print(f"  CEM maxnorm:     {args.mpc_maxnorm}")
    print(f"  Parallel batch:  {args.mpc_rollout_parallel or args.mpc_samples} (of {args.mpc_samples} samples)")
    print(f"  Seed:            {args.seed}")
    print(f"  Env kwargs:      {env_kwargs}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 60)

    # =================================================================
    # 1. Load V-JEPA2 model
    # =================================================================
    print("\n[1/6] Loading V-JEPA2 encoder + predictor...")

    encoder, predictor = init_video_model(
        device=DEVICE,
        patch_size=16,
        max_num_frames=512,
        tubelet_size=2,
        model_name=mcfg["model_name"],
        crop_size=args.image_size,
        pred_depth=mcfg["pred_depth"],
        pred_num_heads=mcfg["pred_num_heads"],
        pred_embed_dim=mcfg["pred_embed_dim"],
        uniform_power=True,
        use_sdpa=True,
        use_rope=True,
        use_silu=False,
        use_pred_silu=False,
        wide_silu=True,
        pred_is_frame_causal=True,
        use_activation_checkpointing=False,
        action_embed_dim=7,
        use_extrinsics=False,
    )

    _load_module_from_ckpt(
        encoder, encoder_ckpt,
        preferred_keys=["target_encoder", "encoder", "model"],
        module_name="encoder",
    )
    _load_module_from_ckpt(
        predictor, predictor_ckpt,
        preferred_keys=["predictor", "model"],
        module_name="predictor",
    )

    encoder = encoder.to(DEVICE).eval()
    predictor = predictor.to(DEVICE).eval()

    # =================================================================
    # 2. Load view embeddings & rebuild predictor attention mask
    # =================================================================
    print("\n[2/6] Setting up multi-view: loading view_embed & rebuilding attention mask...")

    # Load view_embed from the predictor checkpoint
    pred_checkpoint = robust_checkpoint_loader(predictor_ckpt, map_location=torch.device("cpu"))
    view_embed = None
    if "view_embed" in pred_checkpoint:
        ckpt_ve = pred_checkpoint["view_embed"]  # (N_ckpt, 1, D)
        ckpt_camera_views = pred_checkpoint.get("camera_views", None)
        if ckpt_camera_views is not None:
            # Name-based matching: reorder checkpoint view_embed to match --camera-names
            ckpt_view_to_idx = {name: i for i, name in enumerate(ckpt_camera_views)}
            reordered = []
            for name in args.camera_names:
                if name not in ckpt_view_to_idx:
                    raise ValueError(
                        f"Camera '{name}' not found in checkpoint's camera_views "
                        f"{ckpt_camera_views}. Cannot load view_embed."
                    )
                reordered.append(ckpt_ve[ckpt_view_to_idx[name]])
            view_embed = torch.stack(reordered, dim=0).to(DEVICE)
            print(f"  Loaded view_embed by name: checkpoint cameras={ckpt_camera_views}, "
                  f"inference cameras={args.camera_names}, shape={view_embed.shape}")
        else:
            # Legacy checkpoint without camera_views metadata — use positional order
            assert ckpt_ve.shape[0] == num_views, (
                f"view_embed has {ckpt_ve.shape[0]} views but {num_views} cameras specified, "
                f"and checkpoint has no camera_views metadata to match by name"
            )
            view_embed = ckpt_ve.to(DEVICE)
            print(f"  Loaded view_embed (positional, no camera_views in ckpt): "
                  f"shape={view_embed.shape}, device={DEVICE}")
        del ckpt_ve
    else:
        print("  WARNING: No view_embed found in predictor checkpoint. "
              "Proceeding without view embeddings.")
    del pred_checkpoint

    # Rebuild predictor attention mask for multi-view token count.
    # The predictor was trained with grid_height = orig_gh * num_views
    # (see train.py multi-view setup).
    orig_gh = predictor.grid_height
    if num_views > 1:
        # Use a small grid_depth for the mask (we only need 2-3 frames in MPC)
        # to avoid huge memory allocation. The mask is causal so this is fine
        # as long as grid_depth >= frames we'll use (context + rollout).
        mpc_grid_depth = 8  # generous for MPC use
        use_extrinsics = False
        add_tokens = 3 if use_extrinsics else 2

        predictor.grid_height = orig_gh * num_views
        mv_attn_mask = build_action_block_causal_attention_mask(
            mpc_grid_depth,
            predictor.grid_height,
            predictor.grid_width,
            add_tokens=add_tokens,
        )
        predictor.attn_mask = mv_attn_mask
        print(f"  Rebuilt predictor attention mask for multi-view: "
              f"grid_height {orig_gh} → {predictor.grid_height}, "
              f"grid_depth={mpc_grid_depth}, mask shape={mv_attn_mask.shape}")
    else:
        print("  Single view — no attention mask rebuild needed.")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 1.35),
        random_resize_scale=(1.777, 1.777),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.image_size,
    )

    # Build multi-view encoder function
    encode_fn = make_multiview_encoder(
        encoder=encoder,
        transform=transform,
        camera_names=args.camera_names,
        view_embed=view_embed,
        normalize_reps=True,
        device=DEVICE,
    )

    # Build world model (tokens_per_frame is now multi-view)
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout":                args.mpc_rollout,
            "samples":                args.mpc_samples,
            "samples_parallel":       args.mpc_rollout_parallel,
            "topk":                   args.mpc_topk,
            "cem_steps":              args.mpc_cem_steps,
            "momentum_mean":          args.mpc_momentum_mean,
            "momentum_mean_gripper":  args.mpc_momentum_mean_gripper,
            "momentum_std":           args.mpc_momentum_std,
            "momentum_std_gripper":   args.mpc_momentum_std_gripper,
            "maxnorm":                args.mpc_maxnorm,
            "verbose":                True,
        },
        normalize_reps=True,
        device=DEVICE,
    )
    print("  Model loaded.\n")

    # =================================================================
    # 3. Expert rollout
    # =================================================================
    print("[3/6] Running expert rollout...")

    expert_env = make_env(
        args.task, args.seed, args.camera_names, args.image_size,
        max_episode_steps=args.episode_length + 20,
        env_kwargs=env_kwargs,
    )
    expert_frames, expert_render_frames, expert_proprios, expert_rewards = rollout_expert(
        expert_env, args.task, args.episode_length, args.render_camera
    )
    expert_env.close()

    actual_len = len(expert_frames)
    print(f"  Expert rollout done: {actual_len} steps.\n")

    # =================================================================
    # 4. Sample intermediate goal frames
    # =================================================================
    print("[4/6] Sampling intermediate goal frames...")

    if actual_len <= 0:
        raise ValueError("Expert rollout has zero frames; cannot sample goals.")

    if args.n_goals == -1:
        # Frame-by-frame tracking: each expert frame becomes a goal.
        goal_step_indices = list(range(actual_len))
    elif args.n_goals <= 0:
        raise ValueError(
            f"Invalid --n-goals={args.n_goals}. Use -1 (every frame) or a positive integer."
        )
    else:
        n_goals = min(args.n_goals, actual_len)
        goal_step_indices = [
            int(round((i + 1) * actual_len / n_goals)) - 1
            for i in range(n_goals)
        ]
        goal_step_indices[-1] = actual_len - 1

    goal_frames = [expert_frames[idx] for idx in goal_step_indices]
    goal_proprios = [expert_proprios[idx] for idx in goal_step_indices]

    for i, idx in enumerate(goal_step_indices):
        ee = goal_proprios[i][:3]
        print(f"  Goal {i}: expert step {idx:4d} | "
              f"EE=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f})")

    # Save goal frame images for reference (first camera only)
    goal_img_dir = os.path.join(args.output_dir, "goal_frames")
    os.makedirs(goal_img_dir, exist_ok=True)
    for i, gf in enumerate(goal_frames):
        cam_img = gf[:, :, :3]
        Image.fromarray(cam_img).save(
            os.path.join(goal_img_dir, f"goal_{i}_step{goal_step_indices[i]:04d}.png")
        )
    print()

    # =================================================================
    # 5. MPC closed-loop rollout
    # =================================================================
    print("[5/6] Running MPC closed-loop rollout (multi-view)...")

    mpc_env = make_env(
        args.task, args.seed, args.camera_names, args.image_size,
        max_episode_steps=args.max_total_steps + 20,
        env_kwargs=env_kwargs,
    )

    mpc_render_frames, mpc_proprios, mpc_log = run_mpc(
        env=mpc_env,
        world_model=world_model,
        encode_fn=encode_fn,
        tokens_per_frame=tokens_per_frame,
        goal_frames=goal_frames,
        expert_proprios=goal_proprios,
        goal_timesteps=goal_step_indices,
        render_camera_name=args.render_camera,
        max_steps_per_goal=args.max_steps_per_goal,
        max_total_steps=args.max_total_steps,
        goal_rep_threshold=args.goal_rep_threshold,
        warmup_steps=5,
        device=DEVICE,
    )
    mpc_env.close()
    print()

    # =================================================================
    # 6. Visualisation & analysis
    # =================================================================
    print("[6/6] Generating visualisations...")

    # Side-by-side GIF
    gif_path = os.path.join(args.output_dir, "expert_vs_mpc.gif")
    make_side_by_side_gif(
        list(expert_render_frames),
        list(mpc_render_frames),
        gif_path,
        fps=args.gif_fps,
    )

    # 3D trajectory plot
    traj_path = os.path.join(args.output_dir, "ee_trajectories_3d.png")
    make_trajectory_plot(
        expert_proprios, mpc_proprios, goal_proprios,
        goal_step_indices, traj_path
    )

    # Analysis plot
    analysis_path = os.path.join(args.output_dir, "mpc_analysis.png")
    make_analysis_plot(mpc_log, analysis_path)

    # Save log as JSON
    log_path = os.path.join(args.output_dir, "mpc_log.json")
    with open(log_path, "w") as f:
        json.dump(mpc_log, f, indent=2)
    print(f"  Saved analysis log → {log_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (MULTI-VIEW)")
    print("=" * 60)
    print(f"  Task:              {args.task}")
    print(f"  Camera views:      {args.camera_names}")
    print(f"  Expert steps:      {actual_len}")
    print(f"  MPC total steps:   {mpc_log['total_steps']}")
    print(f"  Goals reached:     {mpc_log['goals_reached']}/{mpc_log['n_goals']}")
    if mpc_log["steps"]:
        final = mpc_log["steps"][-1]
        print(f"  Final rep dist:    {final['rep_dist']:.4f}")
        print(f"  Final EE dist:     {final['ee_dist']:.4f} m")
    print(f"  Output dir:        {args.output_dir}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()