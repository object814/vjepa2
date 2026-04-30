#!/usr/bin/env python3
"""
Multi-View Action Inference & Closed-Loop MPC Evaluation.

Multi-camera variant of action_inference.py. The predictor was trained with
multiple camera views and learnable view embeddings.

Key features vs. single-view action_inference.py:
  - All observation cameras are encoded jointly (interleaved per-timestep).
  - Learnable view_embed loaded from the predictor checkpoint.
  - Predictor attention mask rebuilt for multi-view token count.
  - Configurable actions: the user supplies one or more 4D actions, each
    defining a constant-action ground-truth rollout. For each action:
      1. Roll out the GT trajectory in the env.
      2. Visualise the energy landscape (predictor loss over sampled actions).
      3. Run closed-loop MPC toward the goal (last GT frame representation).
      4. Produce a side-by-side GIF (GT left, MPC right) and trajectory plot.
  - All checkpoint paths, task, cameras, and hyperparameters are CLI-configurable.
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

import torch
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
import metaworld  # noqa: F401
from metaworld.wrappers import ProprioMultiImageObsWrapper

# MetaWorld action_scale rescaling (see sawyer_xyz_env.py set_xyz_action)
MW_ACTION_SCALE = 1.0 / 80


# =====================================================================
# Helpers
# =====================================================================

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
    """Burn a text label into the top of an RGB frame."""
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


def parse_actions(actions_str):
    """
    Parse a JSON string of actions.

    Accepts:
      - A single action:   "[0.02, 0.03, -0.01, 0.3]"
      - Multiple actions:  "[[0.02, 0.03, -0.01, 0.3], [0.0, 0.0, 0.2, 0.0]]"

    Returns a list of 4D numpy arrays.
    """
    parsed = json.loads(actions_str)
    if isinstance(parsed[0], (int, float)):
        # Single action: [x, y, z, g]
        parsed = [parsed]
    actions = []
    for a in parsed:
        assert len(a) == 4, f"Each action must have 4 elements [dx, dy, dz, gripper], got {len(a)}: {a}"
        actions.append(np.array(a, dtype=np.float32))
    return actions


# =====================================================================
# Multi-view encoding
# =====================================================================

def forward_target_multiview(camera_tensors, encoder, view_embed,
                             tokens_per_frame_single, normalize_reps, device):
    """
    Encode all camera views for a full trajectory, returning both the raw
    target representation (for loss/goals) and the predictor-input representation
    (with view embeddings).

    Args:
        camera_tensors: list of (1, C, T, H, W) tensors, one per camera view
        encoder: frozen encoder
        view_embed: (num_views, 1, D) or None
        tokens_per_frame_single: H*W patches for a single view
        normalize_reps: whether to apply LayerNorm

    Returns:
        h_target: (1, T * num_views * n_patches, D) — no view embed (for loss/goal)
        h_pred:   (1, T * num_views * n_patches, D) — with view embed (predictor input)
    """
    num_views = len(camera_tensors)
    h_views = []

    for cam in camera_tensors:
        B, C, T, H, W = cam.size()
        c = cam.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        c = c.to(device, non_blocking=True)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1))  # (1, T, n_patches, D)
        h_views.append(h)

    # Raw target: interleave views per-timestep → (1, T, num_views*n_patches, D) → flatten
    h_target = torch.stack(h_views, dim=2).flatten(2, 3).flatten(1, 2)
    if normalize_reps:
        h_target = F.layer_norm(h_target, (h_target.size(-1),))

    # Predictor input: add view embeddings before interleaving
    if num_views > 1 and view_embed is not None:
        h_views_ve = [
            hv + view_embed[i].unsqueeze(0).unsqueeze(0)
            for i, hv in enumerate(h_views)
        ]
        h_pred = torch.stack(h_views_ve, dim=2).flatten(2, 3).flatten(1, 2)
        if normalize_reps:
            h_pred = F.layer_norm(h_pred, (h_pred.size(-1),))
    else:
        h_pred = h_target

    return h_target, h_pred


def encode_frame_multiview(image_np, encoder, camera_names, view_embed,
                           tokens_per_frame_single, normalize_reps, device,
                           for_predictor=False, transform=None):
    """
    Encode a single multi-camera observation (H, W, 3*N_cam) into multi-view
    latent representation.

    Args:
        image_np: (H, W, 3*N_cam) uint8 array
        for_predictor: if True, add view embeddings
        transform: video transform (normalization + spatial crop); required for
                   correct results — raw uint8 pixels must be normalised to
                   match the encoder's training distribution.

    Returns:
        h: (1, num_views * n_patches, D)
    """
    num_views = len(camera_names)
    h_views = []

    for cam_idx in range(num_views):
        cam_image = image_np[:, :, 3 * cam_idx: 3 * (cam_idx + 1)]
        if transform is not None:
            clip = np.expand_dims(cam_image, axis=0)   # (1, H, W, 3)
            cam_t = transform(clip)                     # (C, 1, H, W)
            cam_t = cam_t.unsqueeze(0).to(device)       # (1, C, 1, H, W)
        else:
            cam_t = torch.from_numpy(cam_image).float().to(device)
            cam_t = cam_t.permute(2, 0, 1).unsqueeze(1).unsqueeze(0)  # (1, 3, 1, H, W)
        B, C, T_enc, H_enc, W_enc = cam_t.size()
        c = cam_t.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T_enc, -1, h.size(-1))  # (1, 1, n_patches, D)
        h_views.append(h)

    if for_predictor and view_embed is not None and num_views > 1:
        h_views_ve = [
            hv + view_embed[i].unsqueeze(0).unsqueeze(0)
            for i, hv in enumerate(h_views)
        ]
        h = torch.stack(h_views_ve, dim=2).flatten(2, 3)
    else:
        h = torch.stack(h_views, dim=2).flatten(2, 3)

    h = h.flatten(1, 2)  # (1, num_views * n_patches, D)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


# =====================================================================
# Forward actions through predictor (energy landscape)
# =====================================================================

def forward_actions_multiview(z_pred, states, predictor, tokens_per_frame,
                              nsamples, grid_size, normalize_reps, device,
                              action_repeat=1):
    """
    Forward pass through the predictor for a grid of sampled 3D actions.
    Uses joint multi-view representations.

    Args:
        z_pred: (1, T * tokens_per_frame, D) — predictor-input reps (with view embed)
        states: (1, T, 7)
        tokens_per_frame: num_views * single_view_tokens

    Returns:
        z_hat, s_hat, a_hat — predicted reps, states, and action grids
    """
    def make_action_grid():
        action_samples = []
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples.append(
                        torch.tensor([da, db, dc, 0, 0, 0, 0], device=device, dtype=z_pred.dtype)
                    )
        return torch.stack(action_samples, dim=0).unsqueeze(1)

    action_samples = make_action_grid()
    n_total = int(nsamples ** 3)

    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s

    # Context frame rep and pose
    z_hat = z_pred[:, :tokens_per_frame].repeat(n_total, 1, 1)
    s_hat = states[:, :1].repeat(n_total, 1, 1)
    a_hat = action_samples

    for _ in range(action_repeat):
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    return z_hat, s_hat, a_hat


def loss_fn(z, h, tokens_per_frame):
    """L1 loss between predicted and target reps (last frame only)."""
    z_last = z[:, -tokens_per_frame:]
    h_last = h[:, -tokens_per_frame:]
    loss = torch.mean(torch.abs(z_last - h_last), dim=[1, 2])
    return loss.tolist()


# =====================================================================
# Environment
# =====================================================================

def make_env(task_name, camera_names, image_size, seed=None,
             max_episode_steps=250, env_kwargs=None):
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
# Collect GT rollout
# =====================================================================

def collect_gt_rollout(env, action_4d, T, camera_names, warmup_steps=5,
                       render_camera="corner"):
    """
    Roll out a constant action for T steps, recording observations.

    Returns:
        frames:       list of (H, W, 3*N_cam) observation images
        render_frames: list of rendered RGB frames
        states:       list of (7,) proprioception vectors
        actions:      list of (4,) actions
        initial_ee:   (3,) starting EE position
    """
    obs, _ = env.reset()
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    initial_ee = obs["proprio"][:3].copy()
    frames, render_frames, states, actions = [], [], [], []
    render_frames.append(env.render(camera_name=render_camera).copy())

    for t in range(T):
        obs, _, terminated, truncated, _ = env.step(action_4d)
        frames.append(obs["image"])
        render_frames.append(env.render(camera_name=render_camera).copy())
        states.append(obs["proprio"].copy())
        actions.append(action_4d.copy())
        if terminated or truncated:
            break

    return frames, render_frames, states, actions, initial_ee


def frames_to_camera_tensors(frames, camera_names, device, transform=None):
    """
    Convert list of (H, W, 3*N_cam) frames into per-camera tensors.

    Args:
        transform: video transform (normalization + spatial crop); when provided,
                   each camera's clip is normalised to match the encoder's
                   training distribution.

    Returns:
        camera_tensors: list of (1, C=3, T, H, W) float tensors
    """
    stacked = np.stack(frames, axis=0)  # (T, H, W, 3V)
    num_cameras = len(camera_names)
    camera_tensors = []
    for cam_idx in range(num_cameras):
        cam = stacked[:, :, :, 3 * cam_idx: 3 * (cam_idx + 1)]  # (T, H, W, 3)
        if transform is not None:
            cam_t = transform(cam)               # (C, T, H, W)
        else:
            cam_t = torch.from_numpy(cam).float()
            cam_t = cam_t.permute(0, 3, 1, 2)   # (T, 3, H, W)
            cam_t = cam_t.permute(1, 0, 2, 3)   # (3, T, H, W)
        cam_t = cam_t.unsqueeze(0).to(device)    # (1, C, T, H, W)
        camera_tensors.append(cam_t)
    return camera_tensors


# =====================================================================
# Visualisation
# =====================================================================

def make_side_by_side_gif(gt_frames, mpc_frames, output_path, fps=10,
                          gt_label="Ground Truth", mpc_label="MPC"):
    max_len = max(len(gt_frames), len(mpc_frames))
    while len(gt_frames) < max_len:
        gt_frames.append(gt_frames[-1].copy())
    while len(mpc_frames) < max_len:
        mpc_frames.append(mpc_frames[-1].copy())

    gif_frames = []
    for t in range(max_len):
        left = add_label(gt_frames[t], gt_label)
        right = add_label(mpc_frames[t], mpc_label)
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
        output_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"  Saved GIF ({len(gif_frames)} frames) → {output_path}")


def make_trajectory_plot(real_ee, mpc_ee, goal_ee, start_ee, action_label,
                         output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(real_ee[:, 0], real_ee[:, 1], real_ee[:, 2],
            "b-o", label="Ground Truth", markersize=3, linewidth=1.5, alpha=0.7)
    ax.plot(mpc_ee[:, 0], mpc_ee[:, 1], mpc_ee[:, 2],
            "r-s", label="MPC (multi-view)", markersize=3, linewidth=1.5, alpha=0.7)

    ax.scatter(*start_ee, c="lime", s=150, marker="^", zorder=5,
               edgecolors="k", label="Start")
    ax.scatter(*goal_ee, c="gold", s=200, marker="*", zorder=5,
               edgecolors="k", label="Goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"EE Trajectories: GT vs MPC | action={action_label}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 3D plot → {output_path}")


def make_energy_heatmap(plot_data, nsamples, action_label, output_path):
    delta_x = [d[0] for d in plot_data]
    delta_z = [d[2] for d in plot_data]
    energy  = [d[3] for d in plot_data]

    heatmap, xedges, yedges = np.histogram2d(
        delta_x, delta_z, weights=energy, bins=nsamples
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heatmap.T, origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Action Delta x")
    ax.set_ylabel("Action Delta z")
    ax.set_title(f"Energy Landscape (multi-view) | action={action_label}")
    fig.colorbar(im, ax=ax)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap → {output_path}")


# =====================================================================
# Combined summary visualisations across all actions
# =====================================================================

def make_summary_trajectory_plot(all_results, output_path):
    """
    Single plot showing GT + MPC trajectories for every action, colour-coded.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    for i, res in enumerate(all_results):
        color = cmap(i % 10)
        label_suffix = res["action_label"]
        ax.plot(res["real_ee"][:, 0], res["real_ee"][:, 1], res["real_ee"][:, 2],
                "-o", color=color, markersize=2, linewidth=1.5, alpha=0.6,
                label=f"GT [{label_suffix}]")
        ax.plot(res["mpc_ee"][:, 0], res["mpc_ee"][:, 1], res["mpc_ee"][:, 2],
                "--s", color=color, markersize=2, linewidth=1.5, alpha=0.6,
                label=f"MPC [{label_suffix}]")

        ax.scatter(*res["goal_ee"], c=[color], s=120, marker="*", zorder=5,
                   edgecolors="k")

    # Start point (same for all if same seed)
    if all_results:
        ax.scatter(*all_results[0]["start_ee"], c="lime", s=150, marker="^",
                   zorder=5, edgecolors="k", label="Start")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("EE Trajectories: All Actions Summary (Multi-View)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary trajectory plot → {output_path}")


def make_summary_gif(all_results, output_path, fps=10):
    """
    Combined GIF: each column is one action, top row = GT, bottom row = MPC.
    If too many actions, just do side-by-side per action in a grid.
    """
    n_actions = len(all_results)
    if n_actions == 0:
        return

    # Determine max length across all rollouts
    max_len = 0
    for res in all_results:
        max_len = max(max_len, len(res["gt_render_frames"]), len(res["mpc_render_frames"]))

    # Pad all to max_len
    for res in all_results:
        while len(res["gt_render_frames"]) < max_len:
            res["gt_render_frames"].append(res["gt_render_frames"][-1].copy())
        while len(res["mpc_render_frames"]) < max_len:
            res["mpc_render_frames"].append(res["mpc_render_frames"][-1].copy())

    gif_frames = []
    for t in range(max_len):
        columns = []
        for res in all_results:
            gt_f = add_label(res["gt_render_frames"][t],
                             f"GT {res['action_label']}", position="top")
            mpc_f = add_label(res["mpc_render_frames"][t],
                              f"MPC {res['action_label']}", position="top")
            # Stack GT on top of MPC
            min_w = min(gt_f.shape[1], mpc_f.shape[1])
            if gt_f.shape[1] != min_w:
                pil = Image.fromarray(gt_f)
                new_h = int(gt_f.shape[0] * min_w / gt_f.shape[1])
                gt_f = np.array(pil.resize((min_w, new_h), Image.LANCZOS))
            if mpc_f.shape[1] != min_w:
                pil = Image.fromarray(mpc_f)
                new_h = int(mpc_f.shape[0] * min_w / mpc_f.shape[1])
                mpc_f = np.array(pil.resize((min_w, new_h), Image.LANCZOS))
            col = np.concatenate([gt_f, mpc_f], axis=0)
            columns.append(col)

        # Concat columns side by side
        min_h = min(c.shape[0] for c in columns)
        resized = []
        for c in columns:
            if c.shape[0] != min_h:
                pil = Image.fromarray(c)
                new_w = int(c.shape[1] * min_h / c.shape[0])
                c = np.array(pil.resize((new_w, min_h), Image.LANCZOS))
            resized.append(c)
        frame = np.concatenate(resized, axis=1)
        gif_frames.append(Image.fromarray(frame))

    gif_frames[0].save(
        output_path, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"  Saved summary GIF ({len(gif_frames)} frames, {n_actions} actions) → {output_path}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="V-JEPA2 Multi-View Action Inference & Closed-Loop MPC"
    )
    # --- Task ---
    p.add_argument("--task", type=str, default="pick-place-v3",
                   help="Metaworld task name")
    p.add_argument("--episode-length", type=int, default=40,
                   help="Number of env steps per GT rollout")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")
    p.add_argument("--camera-names", nargs="+",
                   default=["topview", "back", "gripperPOV"],
                   help="Observation cameras (order must match training)")
    p.add_argument("--render-camera", type=str, default="corner",
                   help="Camera for rendering output GIFs")
    p.add_argument("--image-size", type=int, default=224,
                   help="Observation image size")
    p.add_argument("--env-kwargs", nargs="*", default=[],
                   help="Extra env kwargs as key=value")

    # --- Actions ---
    p.add_argument("--actions", type=str, required=True,
                   help='JSON string of action(s). Single: "[0.02,0.03,-0.01,0.3]". '
                        'Multiple: "[[0.02,0.03,-0.01,0.3],[0.0,0.0,0.2,0.0]]". '
                        'Each action is [dx, dy, dz, gripper] and will be applied '
                        'as a constant action for --episode-length steps to create '
                        'a GT trajectory, then MPC tries to reproduce it.')

    # --- Model ---
    p.add_argument("--model", type=str, default="large",
                   choices=["giant", "large"],
                   help="Encoder backbone size")
    p.add_argument("--encoder-ckpt", type=str, default=None,
                   help="Override encoder checkpoint path")
    p.add_argument("--predictor-ckpt", type=str, default=None,
                   help="Override predictor checkpoint path")

    # --- Energy landscape ---
    p.add_argument("--energy-nsamples", type=int, default=5,
                   help="Samples per axis for the energy landscape grid (total = n^3)")
    p.add_argument("--energy-grid-size", type=float, default=0.01,
                   help="Action range for the energy landscape grid")
    p.add_argument("--energy-action-repeat", type=int, default=1,
                   help="Number of action repeats in energy landscape forward")

    # --- MPC / CEM ---
    p.add_argument("--mpc-rollout", type=int, default=2)
    p.add_argument("--mpc-samples", type=int, default=500)
    p.add_argument("--mpc-topk", type=int, default=10)
    p.add_argument("--mpc-cem-steps", type=int, default=15)
    p.add_argument("--mpc-momentum-mean", type=float, default=0.15)
    p.add_argument("--mpc-momentum-mean-gripper", type=float, default=0.15)
    p.add_argument("--mpc-momentum-std", type=float, default=0.75)
    p.add_argument("--mpc-momentum-std-gripper", type=float, default=0.15)
    p.add_argument("--mpc-maxnorm", type=float, default=0.1)
    p.add_argument("--max-mpc-steps", type=int, default=100,
                   help="Max closed-loop MPC steps per action rollout")
    p.add_argument("--goal-threshold", type=float, default=0.01,
                   help="EE distance threshold (metres) to consider goal reached")

    # --- Output ---
    p.add_argument("--output-dir", type=str,
                   default="./output_inference_multiview",
                   help="Directory for output files")
    p.add_argument("--gif-fps", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    num_views = len(args.camera_names)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse user-supplied actions
    user_actions = parse_actions(args.actions)
    n_actions = len(user_actions)

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
            "predictor_ckpt": "/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_multiview/e200.pt",
            "pred_depth": 12,
            "pred_num_heads": 12,
            "pred_embed_dim": 384,
        },
    }
    mcfg = MODEL_CONFIGS[args.model]
    encoder_ckpt = args.encoder_ckpt or mcfg["encoder_ckpt"]
    predictor_ckpt = args.predictor_ckpt or mcfg["predictor_ckpt"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse env kwargs
    env_kwargs = {}
    for kv in args.env_kwargs:
        if "=" not in kv:
            raise ValueError(f"Invalid env kwarg: {kv}")
        k, v = kv.split("=", 1)
        env_kwargs[k] = v

    # Token counts
    single_view_tokens = int((args.image_size // 16) ** 2)
    tokens_per_frame = num_views * single_view_tokens

    print("=" * 60)
    print("V-JEPA2  MULTI-VIEW  ACTION INFERENCE")
    print("=" * 60)
    print(f"  Task:            {args.task}")
    print(f"  Model:           {args.model} ({mcfg['model_name']})")
    print(f"  Device:          {DEVICE}")
    print(f"  Cameras:         {args.camera_names} ({num_views} views)")
    print(f"  Tokens/frame:    {tokens_per_frame} ({num_views}×{single_view_tokens})")
    print(f"  Episode length:  {args.episode_length}")
    print(f"  Actions ({n_actions}):")
    for i, a in enumerate(user_actions):
        print(f"    [{i}] {a.tolist()}")
    print(f"  Encoder ckpt:    {encoder_ckpt}")
    print(f"  Predictor ckpt:  {predictor_ckpt}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 60)

    # =================================================================
    # 1. Load model
    # =================================================================
    print("\n[1] Loading V-JEPA2 encoder + predictor...")

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
    # 2. Load view embeddings & rebuild attention mask
    # =================================================================
    print("\n[2] Setting up multi-view...")

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
            print(f"  Loaded view_embed (positional, no camera_views in ckpt): shape={view_embed.shape}")
        del ckpt_ve
    else:
        print("  WARNING: No view_embed in checkpoint.")
    del pred_checkpoint

    if num_views > 1:
        orig_gh = predictor.grid_height
        add_tokens = 2  # no extrinsics
        predictor.grid_height = orig_gh * num_views
        mv_attn_mask = build_action_block_causal_attention_mask(
            8, predictor.grid_height, predictor.grid_width, add_tokens=add_tokens,
        )
        predictor.attn_mask = mv_attn_mask
        print(f"  Rebuilt attention mask: grid_height {orig_gh}→{predictor.grid_height}, "
              f"mask shape={mv_attn_mask.shape}")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 1.35),
        random_resize_scale=(1.777, 1.777),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.image_size,
    )

    # Build WorldModel for closed-loop MPC
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout":               args.mpc_rollout,
            "samples":               args.mpc_samples,
            "topk":                  args.mpc_topk,
            "cem_steps":             args.mpc_cem_steps,
            "momentum_mean":         args.mpc_momentum_mean,
            "momentum_mean_gripper": args.mpc_momentum_mean_gripper,
            "momentum_std":          args.mpc_momentum_std,
            "momentum_std_gripper":  args.mpc_momentum_std_gripper,
            "maxnorm":               args.mpc_maxnorm,
            "verbose":               True,
        },
        normalize_reps=True,
        device=DEVICE,
    )

    # =================================================================
    # 3. Process each action
    # =================================================================
    all_results = []

    for action_idx, action_4d in enumerate(user_actions):
        action_label = f"({action_4d[0]:.3f},{action_4d[1]:.3f},{action_4d[2]:.3f},{action_4d[3]:.2f})"
        action_dir = os.path.join(args.output_dir, f"action_{action_idx}")
        os.makedirs(action_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ACTION {action_idx}/{n_actions-1}: {action_4d.tolist()}")
        print(f"{'='*60}")

        # ----- 3a. Collect GT rollout -----
        print(f"\n  [3a] Collecting GT rollout (T={args.episode_length})...")
        gt_env = make_env(
            args.task, args.camera_names, args.image_size,
            seed=args.seed, max_episode_steps=args.episode_length + 20,
            env_kwargs=env_kwargs,
        )
        frames, gt_render_frames, gt_states, gt_actions, initial_ee = collect_gt_rollout(
            gt_env, action_4d, args.episode_length, args.camera_names,
            render_camera=args.render_camera,
        )
        gt_env.close()

        actual_T = len(frames)
        real_ee = np.concatenate([initial_ee[np.newaxis, :],
                                   np.array([s[:3] for s in gt_states])], axis=0)
        goal_ee = real_ee[-1]
        print(f"  GT rollout: {actual_T} steps, start={initial_ee}, goal={goal_ee}")

        # Convert to tensors
        camera_tensors = frames_to_camera_tensors(frames, args.camera_names, DEVICE, transform=transform)
        states_t = torch.from_numpy(np.stack(gt_states, axis=0)).float().unsqueeze(0).to(DEVICE)

        # ----- 3b. Forward encode (multi-view) -----
        print(f"\n  [3b] Forward encoding (multi-view)...")
        with torch.no_grad():
            h_target, h_pred = forward_target_multiview(
                camera_tensors, encoder, view_embed,
                single_view_tokens, True, DEVICE,
            )
        print(f"  h_target shape: {h_target.shape}")
        print(f"  h_pred shape:   {h_pred.shape}")

        # ----- 3c. Energy landscape -----
        print(f"\n  [3c] Energy landscape (nsamples={args.energy_nsamples}, grid={args.energy_grid_size})...")
        with torch.no_grad():
            z_hat, s_hat, a_hat = forward_actions_multiview(
                h_pred, states_t, predictor, tokens_per_frame,
                nsamples=args.energy_nsamples,
                grid_size=args.energy_grid_size,
                normalize_reps=True,
                device=DEVICE,
                action_repeat=args.energy_action_repeat,
            )
            loss_vals = loss_fn(z_hat, h_target, tokens_per_frame)

        plot_data = []
        for b, v in enumerate(loss_vals):
            plot_data.append((
                a_hat[b, :-1, 0].sum().detach().cpu().item(),
                a_hat[b, :-1, 1].sum().detach().cpu().item(),
                a_hat[b, :-1, 2].sum().detach().cpu().item(),
                v,
            ))
        heatmap_path = os.path.join(action_dir, "energy_heatmap.png")
        make_energy_heatmap(plot_data, args.energy_nsamples, action_label, heatmap_path)

        # ----- 3d. One-shot CEM planning -----
        print(f"\n  [3d] One-shot CEM planning...")
        with torch.no_grad():
            z_n = h_pred[:, :tokens_per_frame]  # first frame (predictor input)
            z_goal = h_target[:, -tokens_per_frame:]  # last frame (target)
            s_n = states_t[:, :1]
            cem_action = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
            a7 = cem_action[0, 0] if cem_action.ndim == 3 else cem_action[0]
            print(f"  CEM action (world-model space): "
                  f"({a7[0]:+.4f}, {a7[1]:+.4f}, {a7[2]:+.4f}, gripper={a7[6]:+.3f})")

        # ----- 3e. Closed-loop MPC -----
        print(f"\n  [3e] Closed-loop MPC (max {args.max_mpc_steps} steps)...")
        mpc_env = make_env(
            args.task, args.camera_names, args.image_size,
            seed=args.seed, max_episode_steps=args.max_mpc_steps + 20,
            env_kwargs=env_kwargs,
        )
        obs_i, _ = mpc_env.reset()
        for _ in range(5):
            obs_i, _, _, _, _ = mpc_env.step(np.zeros(4, dtype=np.float32))

        # Goal rep (no view embed — raw target)
        z_goal_mpc = h_target[:, -tokens_per_frame:]

        mpc_ee_traj = [obs_i["proprio"][:3].copy()]
        mpc_render_frames = [mpc_env.render(camera_name=args.render_camera).copy()]

        with torch.no_grad():
            for mpc_step in range(args.max_mpc_steps):
                # Encode current (with view embed)
                z_cur = encode_frame_multiview(
                    obs_i["image"], encoder, args.camera_names, view_embed,
                    single_view_tokens, True, DEVICE, for_predictor=True,
                    transform=transform,
                )
                s_cur = (
                    torch.from_numpy(obs_i["proprio"])
                    .float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                )

                mpc_action = world_model.infer_next_action(z_cur, s_cur, z_goal_mpc)
                a7 = mpc_action[0].cpu().numpy()
                if a7.ndim == 2:
                    a7 = a7[0]

                act_4d = np.zeros(4, dtype=np.float32)
                act_4d[:3] = a7[:3] / MW_ACTION_SCALE
                act_4d[3]  = a7[6]

                obs_i, _, terminated, truncated, _ = mpc_env.step(act_4d)
                cur_ee = obs_i["proprio"][:3].copy()
                mpc_ee_traj.append(cur_ee)
                mpc_render_frames.append(
                    mpc_env.render(camera_name=args.render_camera).copy()
                )

                dist = float(np.linalg.norm(cur_ee - goal_ee))
                if mpc_step % 10 == 0 or mpc_step < 5:
                    print(f"    Step {mpc_step+1:3d} | "
                          f"ee=({cur_ee[0]:.4f},{cur_ee[1]:.4f},{cur_ee[2]:.4f}) | "
                          f"dist={dist:.4f}")

                if dist < args.goal_threshold:
                    print(f"    Reached goal (dist={dist:.4f} < {args.goal_threshold})")
                    break
                if terminated or truncated:
                    print("    Episode ended.")
                    break

        mpc_env.close()
        mpc_ee = np.array(mpc_ee_traj)
        print(f"  MPC: {len(mpc_ee_traj)-1} steps executed.")

        # ----- 3f. Per-action visualisations -----
        print(f"\n  [3f] Generating per-action visualisations...")

        gif_path = os.path.join(action_dir, "gt_vs_mpc.gif")
        make_side_by_side_gif(
            list(gt_render_frames), list(mpc_render_frames),
            gif_path, fps=args.gif_fps,
        )

        traj_path = os.path.join(action_dir, "ee_trajectories_3d.png")
        make_trajectory_plot(real_ee, mpc_ee, goal_ee, initial_ee,
                             action_label, traj_path)

        # Collect results for summary
        all_results.append({
            "action_idx":       action_idx,
            "action_4d":        action_4d.tolist(),
            "action_label":     action_label,
            "real_ee":          real_ee,
            "mpc_ee":           mpc_ee,
            "goal_ee":          goal_ee,
            "start_ee":         initial_ee,
            "gt_steps":         actual_T,
            "mpc_steps":        len(mpc_ee_traj) - 1,
            "final_ee_dist":    float(np.linalg.norm(mpc_ee[-1] - goal_ee)),
            "gt_render_frames": list(gt_render_frames),
            "mpc_render_frames": list(mpc_render_frames),
        })

    # =================================================================
    # 4. Summary across all actions
    # =================================================================
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL ACTIONS")
    print(f"{'='*60}")

    summary_traj_path = os.path.join(args.output_dir, "summary_trajectories.png")
    make_summary_trajectory_plot(all_results, summary_traj_path)

    summary_gif_path = os.path.join(args.output_dir, "summary_gt_vs_mpc.gif")
    make_summary_gif(all_results, summary_gif_path, fps=args.gif_fps)

    # Print table
    print(f"\n  {'Idx':>3s}  {'Action':>35s}  {'GT steps':>8s}  {'MPC steps':>9s}  {'Final dist':>10s}")
    print(f"  {'---':>3s}  {'---':>35s}  {'---':>8s}  {'---':>9s}  {'---':>10s}")
    for res in all_results:
        print(f"  {res['action_idx']:3d}  {res['action_label']:>35s}  "
              f"{res['gt_steps']:8d}  {res['mpc_steps']:9d}  "
              f"{res['final_ee_dist']:10.4f} m")

    # Save summary JSON
    summary_data = []
    for res in all_results:
        summary_data.append({
            "action_idx":    res["action_idx"],
            "action_4d":     res["action_4d"],
            "gt_steps":      res["gt_steps"],
            "mpc_steps":     res["mpc_steps"],
            "final_ee_dist": res["final_ee_dist"],
            "start_ee":      res["start_ee"].tolist(),
            "goal_ee":       res["goal_ee"].tolist(),
            "mpc_final_ee":  res["mpc_ee"][-1].tolist(),
        })
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Saved summary → {summary_path}")

    print(f"\n  Output directory: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
