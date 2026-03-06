#!/usr/bin/env python3
"""
MPC Evaluation Script: Expert-Guided Task Completion with V-JEPA2

Pipeline:
  1. Create two identical Metaworld environments (expert + MPC).
  2. Roll out the expert policy, recording observation frames.
  3. Evenly sample K intermediate goal frames from the expert rollout.
  4. Encode each goal frame into latent representations.
  5. Run closed-loop MPC: at each step, plan an action with CEM that moves
     the current latent representation toward the current goal representation.
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
# MetaWorld's `set_xyz_action` clips the incoming action to [-1, 1] and then
# applies  pos_delta = action * action_scale  where action_scale = 1/80.
# The CEM world model plans in *state-delta* space (meters), matching the
# training data:  actions = states[1:] - states[:-1].
# To convert CEM output (meters) → MetaWorld action units we divide by
# action_scale (equivalently, multiply by 80).
MW_ACTION_SCALE = 1.0 / 80          # MetaWorld's SawyerXYZEnv.action_scale
import torch
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader
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


# =====================================================================
# Helpers
# =====================================================================

POLICY_MAP = {
    "pick-place-v3":    SawyerPickPlaceV3Policy,
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

def rollout_expert(env, task_name, episode_length, camera_name, warmup_steps=5):
    """
    Run the scripted expert policy, returning recorded data.

    Returns
    -------
    frames        : list[np.ndarray]  – observation images  (H, W, 3*N_cam)
    render_frames : list[np.ndarray]  – rendered RGB frames from `camera_name`
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
    render_frames.append(env.render(camera_name=camera_name).copy())

    for t in range(episode_length):
        action = policy.get_action(obs["original_obs"])
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["image"])
        render_frames.append(env.render(camera_name=camera_name).copy())
        proprios.append(obs["proprio"].copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    print(f"  Expert rollout: {len(frames)} steps, final reward={rewards[-1]:.3f}")
    return frames, render_frames, proprios, rewards


# =====================================================================
# MPC closed-loop controller
# =====================================================================

@torch.no_grad()
def run_mpc(
    env,
    world_model,
    encoder,
    transform,
    tokens_per_frame,
    goal_frames,          # list of K np.ndarray goal images (H, W, 3*N_cam)
    expert_proprios,      # list of proprios at goal timesteps (for EE logging)
    goal_timesteps,       # list of ints – expert timestep for each goal
    cam_idx,              # which camera channel to use for encoding
    camera_name,          # camera name for rendering
    max_steps_per_goal,
    max_total_steps,
    goal_rep_threshold,
    warmup_steps,
    device,
    normalize_reps=True,
):
    """
    Closed-loop MPC using intermediate goal representations.

    Returns
    -------
    mpc_render_frames : list[np.ndarray]
    mpc_proprios      : list[np.ndarray]
    log               : dict  (per-step analysis data)
    """

    # --- Encode goal frames into latent space ---
    def encode_image(image_np):
        """Encode a single multi-camera observation into latent tokens."""
        cam_image = image_np[:, :, 3 * cam_idx: 3 * (cam_idx + 1)]
        clip = np.expand_dims(cam_image, axis=0)  # (1, H, W, 3)
        clip = transform(clip)[None, :]            # (1, C, 1, H, W)
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(device, non_blocking=True)
        h = encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h  # (1, N_tokens, D)

    goal_reps = [encode_image(gf) for gf in goal_frames]
    n_goals = len(goal_reps)
    print(f"  Encoded {n_goals} goal frames into latent representations.")

    # --- Reset and warm up MPC env ---
    obs, _ = env.reset()
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    # --- Tracking variables ---
    mpc_render_frames = [env.render(camera_name=camera_name).copy()]
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
        # Encode current observation
        z_current = encode_image(obs["image"])
        s_current = (
            torch.from_numpy(obs["proprio"])
            .float().unsqueeze(0).unsqueeze(0).to(device)
        )

        z_goal = goal_reps[current_goal_idx]

        # Compute representation distance to current goal
        rep_dist = torch.mean(torch.abs(
            z_current[:, :tokens_per_frame] - z_goal[:, :tokens_per_frame]
        )).item()

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
        mpc_action_7d = world_model.infer_next_action(
            z_current, s_current, z_goal
        )
        a7 = mpc_action_7d[0].cpu().numpy()  # (rollout, 7) → take first step
        if a7.ndim == 2:
            a7 = a7[0]

        # --- Action-space conversion (world-model → MetaWorld) ----------
        # CEM output is in world-model space (delta meters, from training
        # data where actions = state_diffs).  MetaWorld env.step() expects
        # actions in [-1, 1] which are then scaled by action_scale (1/80).
        # Without rescaling, a CEM output of 0.01 m would produce only
        # 0.01 × action_scale = 0.000125 m actual displacement (80× too small).
        action_4d = np.zeros(4, dtype=np.float32)
        action_4d[:3] = a7[:3] / MW_ACTION_SCALE   # meters → MW action units
        action_4d[3]  = a7[6]                       # gripper (passthrough)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_4d)
        mpc_render_frames.append(env.render(camera_name=camera_name).copy())
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
    # Pad shorter sequence by repeating last frame
    while len(expert_frames) < max_len:
        expert_frames.append(expert_frames[-1].copy())
    while len(mpc_frames) < max_len:
        mpc_frames.append(mpc_frames[-1].copy())

    gif_frames = []
    for t in range(max_len):
        left = add_label(expert_frames[t], "Expert")
        right = add_label(mpc_frames[t], "MPC")
        # Match heights
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

    # Mark goals
    for i, (ge, ts) in enumerate(zip(goal_ee, goal_timesteps)):
        ax.scatter(*ge, c="gold", s=120, marker="*", zorder=5, edgecolors="k",
                   label=f"Goal {i} (t={ts})" if i < 3 else "")

    ax.scatter(*expert_ee[0], c="lime", s=100, marker="^", zorder=5,
               edgecolors="k", label="Start")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EE Trajectories: Expert vs MPC")
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

    # Representation distance
    ax1.plot(steps, rep_dists, "b-", linewidth=1, label="Rep L1 distance")
    ax1.set_ylabel("Representation Distance (L1)")
    ax1.set_title("MPC Analysis: Distances to Current Goal Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # EE distance
    ax2.plot(steps, ee_dists, "r-", linewidth=1, label="EE L2 distance (m)")
    ax2.set_ylabel("EE Distance (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Goal index
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
        description="V-JEPA2 MPC Evaluation: follow expert demonstrations"
    )
    # --- Task ---
    p.add_argument("--task", type=str, default="pick-place-v3",
                   choices=list(POLICY_MAP.keys()),
                   help="Metaworld task name")
    p.add_argument("--episode-length", type=int, default=200,
                   help="Expert episode length (number of env steps)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible environments")
    p.add_argument("--camera-name", type=str, default="front",
                   help="Camera name for observations and rendering")
    p.add_argument("--image-size", type=int, default=224,
                   help="Observation image size (square, in pixels)")
    p.add_argument("--env-kwargs", nargs="*", default=[],
                   help="Extra env kwargs as key=value (e.g. initialise_region=fixed)")

    # --- Model ---
    p.add_argument("--model", type=str, default="giant",
                   choices=["giant", "large"],
                   help="Encoder backbone size")
    p.add_argument("--encoder-ckpt", type=str, default=None,
                   help="Override encoder checkpoint path")
    p.add_argument("--predictor-ckpt", type=str, default=None,
                   help="Override predictor checkpoint path")

    # --- Goals ---
    p.add_argument("--n-goals", type=int, default=5,
                   help="Number of intermediate goal frames to sample from expert rollout")

    # --- MPC / CEM ---
    # Paper reference: V-JEPA 2 (arxiv:2506.09985), Section 4.1, 4.2, 11.2, Table 3.
    # The paper uses: 800 samples, 10 refinement steps, top-10 elites,
    # planning horizon 1, actions constrained to L1-ball of radius 0.075.
    p.add_argument("--mpc-rollout", type=int, default=1,
                   help="CEM planning horizon (steps ahead to simulate). "
                        "Paper §11.2: horizon=1 ('short planning horizon sufficient').")
    p.add_argument("--mpc-samples", type=int, default=800,
                   help="CEM: action trajectories sampled per iteration. "
                        "Paper §11.2/Table 3: 800 samples (16 sec/action on RTX 4090).")
    p.add_argument("--mpc-topk", type=int, default=10,
                   help="CEM: elite samples to fit the next distribution. "
                        "Paper §11.2: top 10.")
    p.add_argument("--mpc-cem-steps", type=int, default=10,
                   help="CEM: optimization / refinement iterations per action. "
                        "Paper §11.2/Table 3: 10 refinement steps.")
    p.add_argument("--mpc-momentum-mean", type=float, default=0.1,
                   help="CEM: momentum for mean update (xyz). "
                        "Paper uses standard CEM (full elite replacement → 0.0). "
                        "Small momentum (0.1) adds smoothing for sim stability.")
    p.add_argument("--mpc-momentum-mean-gripper", type=float, default=0.1,
                   help="CEM: momentum for mean update (gripper)")
    p.add_argument("--mpc-momentum-std", type=float, default=0.5,
                   help="CEM: momentum for std update (xyz). "
                        "Paper uses standard CEM. "
                        "Moderate momentum prevents premature std collapse in sim.")
    p.add_argument("--mpc-momentum-std-gripper", type=float, default=0.1,
                   help="CEM: momentum for std update (gripper)")
    p.add_argument("--mpc-maxnorm", type=float, default=0.0125,
                   help="CEM: per-axis action magnitude clip (in meters, world-model space). "
                        "Paper §4.1 uses 0.075 for Droid (real Franka). "
                        "For MetaWorld, set to match action_scale = 1/80 = 0.0125 m, "
                        "so CEM-planned actions stay within the env's executable range "
                        "(mw_action = cem_xyz / action_scale ∈ [-1, 1]).")

    # --- Goal switching ---
    # Paper §4.2/§11.2: pick-and-place uses fixed step counts per sub-goal
    # (4 steps for grasp goal, 10 steps for transport goal, 4 steps for place goal).
    # We use representation-distance based switching with a hard step budget fallback.
    p.add_argument("--goal-rep-threshold", type=float, default=0.3,
                   help="Representation L1 distance threshold to switch goals. "
                        "Lower → more precise matching before switching.")
    p.add_argument("--max-steps-per-goal", type=int, default=25,
                   help="Max MPC steps per goal. Paper uses 4-10 steps per sub-goal "
                        "on real Franka at 4fps. Sim may need more steps.")
    p.add_argument("--max-total-steps", type=int, default=200,
                   help="Maximum total MPC steps across all goals")

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="./output_mpc_eval",
                   help="Directory for output files")
    p.add_argument("--gif-fps", type=int, default=15,
                   help="Frames per second for output GIFs")

    return p.parse_args()


def main():
    args = parse_args()

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
            "predictor_ckpt": "/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_0225/e300.pt",
            "pred_depth": 12,
            "pred_num_heads": 12,
            "pred_embed_dim": 384,
        },
    }
    mcfg = MODEL_CONFIGS[args.model]
    encoder_ckpt = args.encoder_ckpt or mcfg["encoder_ckpt"]
    predictor_ckpt = args.predictor_ckpt or mcfg["predictor_ckpt"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    camera_names = [args.camera_name]

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Parse env kwargs ---
    env_kwargs = {}
    for kv in args.env_kwargs:
        if "=" not in kv:
            raise ValueError(f"Invalid env kwarg: {kv}. Expected key=value.")
        k, v = kv.split("=", 1)
        env_kwargs[k] = v

    print("=" * 60)
    print("V-JEPA2  MPC  EVALUATION")
    print("=" * 60)
    print(f"  Task:            {args.task}")
    print(f"  Model:           {args.model} ({mcfg['model_name']})")
    print(f"  Device:          {DEVICE}")
    print(f"  Camera:          {args.camera_name}")
    print(f"  Expert length:   {args.episode_length}")
    print(f"  N goals:         {args.n_goals}")
    print(f"  Max steps/goal:  {args.max_steps_per_goal}")
    print(f"  Max total steps: {args.max_total_steps}")
    print(f"  Goal threshold:  {args.goal_rep_threshold}")
    print(f"  CEM maxnorm:     {args.mpc_maxnorm}")
    print(f"  Seed:            {args.seed}")
    print(f"  Env kwargs:      {env_kwargs}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 60)

    # =================================================================
    # 1. Load V-JEPA2 model
    # =================================================================
    print("\n[1/5] Loading V-JEPA2 encoder + predictor...")

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
    tokens_per_frame = int((args.image_size // encoder.patch_size) ** 2)

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.image_size,
    )

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout":                args.mpc_rollout,
            "samples":                args.mpc_samples,
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
    # 2. Expert rollout
    # =================================================================
    print("[2/5] Running expert rollout...")

    expert_env = make_env(
        args.task, args.seed, camera_names, args.image_size,
        max_episode_steps=args.episode_length + 20,
        env_kwargs=env_kwargs,
    )
    expert_frames, expert_render_frames, expert_proprios, expert_rewards = rollout_expert(
        expert_env, args.task, args.episode_length, args.camera_name
    )
    expert_env.close()

    actual_len = len(expert_frames)
    print(f"  Expert rollout done: {actual_len} steps.\n")

    # =================================================================
    # 3. Sample intermediate goal frames
    # =================================================================
    print("[3/5] Sampling intermediate goal frames...")

    n_goals = min(args.n_goals, actual_len)
    # Evenly spaced indices: e.g. for 200 steps and 5 goals → steps 40,80,120,160,200
    goal_step_indices = [
        int(round((i + 1) * actual_len / n_goals)) - 1
        for i in range(n_goals)
    ]
    # Ensure last goal is the final frame
    goal_step_indices[-1] = actual_len - 1

    goal_frames = [expert_frames[idx] for idx in goal_step_indices]
    goal_proprios = [expert_proprios[idx] for idx in goal_step_indices]

    for i, idx in enumerate(goal_step_indices):
        ee = goal_proprios[i][:3]
        print(f"  Goal {i}: expert step {idx:4d} | "
              f"EE=({ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f})")

    # Save goal frame images for reference
    goal_img_dir = os.path.join(args.output_dir, "goal_frames")
    os.makedirs(goal_img_dir, exist_ok=True)
    for i, gf in enumerate(goal_frames):
        # Take just the first camera's channels for saving
        cam_img = gf[:, :, :3]
        Image.fromarray(cam_img).save(
            os.path.join(goal_img_dir, f"goal_{i}_step{goal_step_indices[i]:04d}.png")
        )
    print()

    # =================================================================
    # 4. MPC closed-loop rollout
    # =================================================================
    print("[4/5] Running MPC closed-loop rollout...")

    mpc_env = make_env(
        args.task, args.seed, camera_names, args.image_size,
        max_episode_steps=args.max_total_steps + 20,
        env_kwargs=env_kwargs,
    )

    mpc_render_frames, mpc_proprios, mpc_log = run_mpc(
        env=mpc_env,
        world_model=world_model,
        encoder=encoder,
        transform=transform,
        tokens_per_frame=tokens_per_frame,
        goal_frames=goal_frames,
        expert_proprios=goal_proprios,
        goal_timesteps=goal_step_indices,
        cam_idx=0,
        camera_name=args.camera_name,
        max_steps_per_goal=args.max_steps_per_goal,
        max_total_steps=args.max_total_steps,
        goal_rep_threshold=args.goal_rep_threshold,
        warmup_steps=5,
        device=DEVICE,
    )
    mpc_env.close()
    print()

    # =================================================================
    # 5. Visualisation & analysis
    # =================================================================
    print("[5/5] Generating visualisations...")

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

    # Analysis plot (rep distance & EE distance over time)
    analysis_path = os.path.join(args.output_dir, "mpc_analysis.png")
    make_analysis_plot(mpc_log, analysis_path)

    # Save log as JSON
    log_path = os.path.join(args.output_dir, "mpc_log.json")
    with open(log_path, "w") as f:
        json.dump(mpc_log, f, indent=2)
    print(f"  Saved analysis log → {log_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Task:              {args.task}")
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
