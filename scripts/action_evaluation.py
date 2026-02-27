"""
Action Evaluation Script
========================
Rolls out a ground-truth trajectory in MetaWorld, and at every timestep
evaluates a uniform 5x5x5 action grid around the current EE position
through the V-JEPA2 predictor.  For each sampled action the predictor
produces a latent next-frame, which is compared to the *goal* latent
(encoded from the last real frame).  The resulting energy landscape is
visualised as 2-D heatmaps (XY, XZ, YZ slices through the cube) with
the ground-truth action direction highlighted.  After visualisation the
ground-truth action is executed so the rollout advances.

Outputs (saved to ./output_eval/):
  - Per-timestep energy heatmaps for each camera
  - Summary 3-D scatter of best-action vs GT-action across time
  - A GIF animating the per-step heatmaps
"""

import sys
sys.path.insert(0, "..")

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path
from PIL import Image

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader
from utils.mpc_utils import compute_new_pose

# ---- Metaworld ----
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper


# ==========================================================
# CONFIG
# ==========================================================
parser = argparse.ArgumentParser(description="V-JEPA2 action evaluation")
parser.add_argument(
    "--model", type=str, default="giant",
    choices=["giant", "large"],
    help="Encoder backbone: 'giant' (vit_giant_xformers) or 'large' (vit_large)",
)
args = parser.parse_args()

# --- Model-dependent config ---
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

MCFG = MODEL_CONFIGS[args.model]

TASK_NAME = "pick-place-v3"
IMAGE_SIZE = 224
T = 40                              # rollout length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NSAMPLES = 10                       # per axis → 1000 total
GRID_SIZE = 0.3                     # ±30 cm around zero
camera_names = ["front"]
GT_ACTION = np.array([0.0, 0.0, -0.2, 0.0], dtype=np.float32)

ENCODER_CKPT = MCFG["encoder_ckpt"]
PREDICTOR_CKPT = MCFG["predictor_ckpt"]

OUTPUT_DIR = "./output_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using model: {args.model} ({MCFG['model_name']})")
print("Using device:", DEVICE)


# ==========================================================
# Helpers
# ==========================================================
def _sanitize_state_dict(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def _find_state_dict(ckpt, preferred):
    for k in preferred:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k], k
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"], "state_dict"
    if isinstance(ckpt, dict) and ckpt:
        first = next(iter(ckpt.values()))
        if torch.is_tensor(first):
            return ckpt, "<root>"
    raise KeyError(f"No state_dict found. Keys: {list(ckpt.keys())}")


def _load_module(module, path, preferred, name):
    ckpt = robust_checkpoint_loader(path, map_location=torch.device("cpu"))
    sd, key = _find_state_dict(ckpt, preferred)
    sd = _sanitize_state_dict(sd)
    msg = module.load_state_dict(sd, strict=False)
    print(f"Loaded {name} from {path} (key='{key}') → {msg}")


def make_env():
    env = gym.make(
        "Meta-World/MT1",
        env_name=TASK_NAME,
        render_mode="rgb_array",
        max_episode_steps=250,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        camera_names=camera_names,
    )
    return env


# ==========================================================
# Encode a single frame (one camera)
# ==========================================================
def encode_single_frame(image_np, cam_idx, encoder, tokens_per_frame, device):
    """
    Encode one camera's RGB frame → (1, tokens_per_frame, D) representation.
    Same encoding path as forward_target to keep representations comparable.
    """
    cam = image_np[:, :, 3 * cam_idx : 3 * (cam_idx + 1)]  # (H, W, 3)
    t = torch.from_numpy(cam).float().to(device)
    t = t.permute(2, 0, 1).unsqueeze(1).unsqueeze(0)        # (1,3,1,H,W)
    B, C, T_, H, W = t.size()
    c = t.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T_, -1, h.size(-1)).flatten(1, 2)
    h = F.layer_norm(h, (h.size(-1),))
    return h


# ==========================================================
# Build the 5×5×5 action grid
# ==========================================================
def make_action_grid(nsamples, grid_size, device, dtype):
    """Return (N^3, 1, 7) tensor with xyz deltas; orientation+gripper = 0."""
    samples = []
    xs = np.linspace(-grid_size, grid_size, nsamples)
    ys = np.linspace(-grid_size, grid_size, nsamples)
    zs = np.linspace(-grid_size, grid_size, nsamples)
    for dx in xs:
        for dy in ys:
            for dz in zs:
                samples.append(torch.tensor([dx, dy, dz, 0, 0, 0, 0],
                                            device=device, dtype=dtype))
    return torch.stack(samples, dim=0).unsqueeze(1), xs, ys, zs


# ==========================================================
# Evaluate energy of every action in the grid
# ==========================================================
def evaluate_action_grid(z_current, s_current, z_goal,
                         predictor, tokens_per_frame,
                         nsamples, grid_size, device):
    """
    Args
        z_current : (1, tokens_per_frame, D) – current frame rep
        s_current : (1, 1, 7) – current proprio state
        z_goal    : (1, tokens_per_frame, D) – goal frame rep
    Returns
        energies  : (N^3,) numpy – L1 energy for each sampled action
        actions_7d: (N^3, 7) numpy – the sampled actions
        xs, ys, zs: 1-D arrays of unique grid values per axis
    """
    N3 = nsamples ** 3
    action_grid, xs, ys, zs = make_action_grid(
        nsamples, grid_size, device, z_current.dtype
    )  # (N^3, 1, 7)

    z_rep = z_current.repeat(N3, 1, 1)      # (N^3, tokens, D)
    s_rep = s_current.repeat(N3, 1, 1)      # (N^3, 1, 7)

    # One-step predictor
    z_pred = predictor(z_rep, action_grid, s_rep)[:, -tokens_per_frame:]
    z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

    # Goal comparison
    g = z_goal.repeat(N3, 1, 1)
    energy = torch.mean(torch.abs(z_pred - g), dim=[1, 2])  # (N^3,)

    return (energy.cpu().numpy(),
            action_grid[:, 0, :].cpu().numpy(),
            xs, ys, zs)


# ==========================================================
# Plotting helpers
# ==========================================================
def _plot_heatmap_slice(energies, actions, ax_i, ax_j, ax_label_i, ax_label_j,
                        gt_action_7d, nsamples, ax):
    """
    Project energies onto a 2-D slice by taking the *minimum* energy along
    the third axis (best case).  Overlay a star at the GT action.
    """
    # Bin into 2-D grid (min-energy projection)
    unique_i = np.unique(actions[:, ax_i])
    unique_j = np.unique(actions[:, ax_j])
    heatmap = np.full((len(unique_i), len(unique_j)), np.inf)
    for idx in range(len(actions)):
        ii = np.searchsorted(unique_i, actions[idx, ax_i])
        jj = np.searchsorted(unique_j, actions[idx, ax_j])
        if ii < len(unique_i) and jj < len(unique_j):
            heatmap[ii, jj] = min(heatmap[ii, jj], energies[idx])

    im = ax.imshow(
        heatmap.T, origin="lower",
        extent=[unique_i[0], unique_i[-1], unique_j[0], unique_j[-1]],
        aspect="auto", cmap="viridis",
    )
    # GT action marker
    ax.plot(gt_action_7d[ax_i], gt_action_7d[ax_j],
            marker="*", color="red", markersize=14, markeredgecolor="white",
            markeredgewidth=1.0, label="GT action")
    # Best action marker
    best_idx = np.argmin(energies)
    ax.plot(actions[best_idx, ax_i], actions[best_idx, ax_j],
            marker="o", color="cyan", markersize=10, markeredgecolor="white",
            markeredgewidth=1.0, label="Best sampled")
    ax.set_xlabel(ax_label_i)
    ax.set_ylabel(ax_label_j)
    ax.legend(fontsize=7, loc="upper right")
    return im


def plot_energy_step(energies, actions, gt_action_7d, step, cam_name,
                     nsamples, output_dir):
    """
    Create a figure with three 2-D slices (XY, XZ, YZ) and save it.
    Returns the path to the saved image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    slices = [(0, 1, "Δx", "Δy"), (0, 2, "Δx", "Δz"), (1, 2, "Δy", "Δz")]
    for ax, (ai, aj, li, lj) in zip(axes, slices):
        im = _plot_heatmap_slice(energies, actions, ai, aj, li, lj,
                                 gt_action_7d, nsamples, ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Energy Landscape – {cam_name} – step {step}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f"energy_step{step:03d}_{cam_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================
# Load model
# ==========================================================
print("Initializing V-JEPA2-AC …")

encoder, predictor = init_video_model(
    device=DEVICE,
    patch_size=16, max_num_frames=512, tubelet_size=2,
    model_name=MCFG["model_name"], crop_size=IMAGE_SIZE,
    pred_depth=MCFG["pred_depth"], pred_num_heads=MCFG["pred_num_heads"],
    pred_embed_dim=MCFG["pred_embed_dim"],
    uniform_power=True, use_sdpa=True, use_rope=True,
    use_silu=False, use_pred_silu=False, wide_silu=True,
    pred_is_frame_causal=True, use_activation_checkpointing=False,
    action_embed_dim=7, use_extrinsics=False,
)

_load_module(encoder, ENCODER_CKPT,
             ["target_encoder", "encoder", "model"], "encoder")
_load_module(predictor, PREDICTOR_CKPT,
             ["predictor", "model"], "predictor")

encoder = encoder.to(DEVICE).eval()
predictor = predictor.to(DEVICE).eval()
tokens_per_frame = int((IMAGE_SIZE // encoder.patch_size) ** 2)


# ==========================================================
# Collect ground-truth rollout
# ==========================================================
print("\n=== Collecting ground-truth rollout ===")

env = make_env()
obs, _ = env.reset(seed=0)

# Warmup (same as action_inference.py)
for _ in range(5):
    obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

gt_images = [obs["image"].copy()]          # list of (H,W,3*N_cam)
gt_proprios = [obs["proprio"].copy()]      # list of (7,)
gt_actions = []                            # list of (4,)
gt_renders = [env.render().copy()]         # for GIF

print("Rolling out GT trajectory …")
for t in range(T):
    obs, _, terminated, truncated, _ = env.step(GT_ACTION)
    gt_images.append(obs["image"].copy())
    gt_proprios.append(obs["proprio"].copy())
    gt_actions.append(GT_ACTION.copy())
    gt_renders.append(env.render().copy())
    if terminated or truncated:
        break

env.close()
actual_T = len(gt_actions)  # may be < T if terminated early
print(f"Collected {actual_T} steps ({actual_T + 1} frames).")


# ==========================================================
# Encode goal frame (last frame) for each camera
# ==========================================================
print("\nEncoding goal frame for each camera …")
camera_z_goals = {}
with torch.no_grad():
    for cam_idx in range(len(camera_names)):
        camera_z_goals[cam_idx] = encode_single_frame(
            gt_images[-1], cam_idx, encoder, tokens_per_frame, DEVICE
        )
        print(f"  {camera_names[cam_idx]}: goal rep shape = "
              f"{camera_z_goals[cam_idx].shape}")


# ==========================================================
# Step-by-step evaluation
# ==========================================================
print("\n=== Step-by-step energy evaluation ===")

# Storage for summary stats
summary = {cam_idx: {"gt_rank": [], "best_action": [], "gt_energy": [],
                      "best_energy": [], "heatmap_paths": []}
           for cam_idx in range(len(camera_names))}

with torch.no_grad():
    for t in range(actual_T):
        # Current observation & GT action for this step
        image_t = gt_images[t]
        proprio_t = gt_proprios[t]
        gt_act_4d = gt_actions[t]
        # Represent GT action in 7-D format: [dx,dy,dz,0,0,0,gripper]
        gt_act_7d = np.array([gt_act_4d[0], gt_act_4d[1], gt_act_4d[2],
                              0, 0, 0, gt_act_4d[3]], dtype=np.float32)

        s_t = (torch.from_numpy(proprio_t).float()
               .unsqueeze(0).unsqueeze(0).to(DEVICE))   # (1,1,7)

        for cam_idx in range(len(camera_names)):
            cam_name = camera_names[cam_idx]

            # Encode current frame
            z_t = encode_single_frame(image_t, cam_idx, encoder,
                                      tokens_per_frame, DEVICE)
            z_goal = camera_z_goals[cam_idx]

            # Evaluate the grid
            energies, actions_7d, xs, ys, zs = evaluate_action_grid(
                z_t, s_t, z_goal,
                predictor, tokens_per_frame,
                NSAMPLES, GRID_SIZE, DEVICE,
            )

            # Also evaluate the GT action itself
            gt_a_tensor = (torch.from_numpy(gt_act_7d).float()
                           .unsqueeze(0).unsqueeze(0).to(DEVICE))  # (1,1,7)
            z_gt_pred = predictor(z_t, gt_a_tensor, s_t)[:, -tokens_per_frame:]
            z_gt_pred = F.layer_norm(z_gt_pred, (z_gt_pred.size(-1),))
            gt_energy = torch.mean(torch.abs(z_gt_pred - z_goal)).item()

            # Rank of GT action energy among grid samples
            rank = int((energies < gt_energy).sum()) + 1  # 1-based
            best_idx = int(np.argmin(energies))

            summary[cam_idx]["gt_rank"].append(rank)
            summary[cam_idx]["gt_energy"].append(gt_energy)
            summary[cam_idx]["best_energy"].append(energies[best_idx])
            summary[cam_idx]["best_action"].append(actions_7d[best_idx, :3].copy())

            # Plot & save
            path = plot_energy_step(energies, actions_7d, gt_act_7d,
                                    t, cam_name, NSAMPLES, OUTPUT_DIR)
            summary[cam_idx]["heatmap_paths"].append(path)

            print(f"  step {t:3d} | {cam_name:>8s} | "
                  f"GT energy={gt_energy:.4f}  best={energies[best_idx]:.4f}  "
                  f"rank={rank}/{len(energies)}")


# ==========================================================
# Summary: GT rank over time  (per camera)
# ==========================================================
print("\n=== Generating summary plots ===")

for cam_idx in range(len(camera_names)):
    cam_name = camera_names[cam_idx]
    ranks = summary[cam_idx]["gt_rank"]
    gt_e = summary[cam_idx]["gt_energy"]
    best_e = summary[cam_idx]["best_energy"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Rank plot
    ax1.bar(range(len(ranks)), ranks, color="steelblue")
    ax1.axhline(1, color="green", linestyle="--", label="Rank 1 (ideal)")
    ax1.set_ylabel("GT action rank among 125 samples")
    ax1.set_title(f"GT Action Rank Over Time – {cam_name}")
    ax1.legend()

    # Energy comparison
    ax2.plot(gt_e, "r-o", markersize=3, label="GT action energy")
    ax2.plot(best_e, "b-s", markersize=3, label="Best sampled energy")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Energy (L1)")
    ax2.set_title(f"Energy Over Time – {cam_name}")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"summary_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ==========================================================
# Summary: 3-D scatter – best sampled vs GT action direction
# ==========================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# GT action is constant here, plot as a single arrow from origin
gt_dir = GT_ACTION[:3]
ax.quiver(0, 0, 0, gt_dir[0], gt_dir[1], gt_dir[2],
          color="blue", linewidth=2.5, label="GT action direction",
          arrow_length_ratio=0.15)

cam_colors = ["red", "green", "magenta", "cyan"]
for cam_idx in range(len(camera_names)):
    cam_name = camera_names[cam_idx]
    bests = np.array(summary[cam_idx]["best_action"])  # (T, 3)
    color = cam_colors[cam_idx % len(cam_colors)]
    ax.scatter(bests[:, 0], bests[:, 1], bests[:, 2],
               c=color, s=20, alpha=0.7, label=f"Best sampled ({cam_name})")

ax.set_xlabel("Δx")
ax.set_ylabel("Δy")
ax.set_zlabel("Δz")
ax.set_title("Best Sampled Action vs GT Action Direction")
ax.legend(loc="best")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "best_vs_gt_3d.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path}")


# ==========================================================
# Animated GIF of per-step heatmaps (one per camera)
# ==========================================================
print("\nCreating animated GIFs …")

for cam_idx in range(len(camera_names)):
    cam_name = camera_names[cam_idx]
    paths = summary[cam_idx]["heatmap_paths"]
    if not paths:
        continue
    pil_frames = [Image.open(p) for p in paths]
    # Resize to match first frame
    w, h = pil_frames[0].size
    pil_frames = [f.resize((w, h), Image.LANCZOS) if f.size != (w, h) else f
                  for f in pil_frames]
    gif_path = os.path.join(OUTPUT_DIR, f"energy_anim_{cam_name}.gif")
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=300, loop=0,
    )
    print(f"  Saved {gif_path}  ({len(pil_frames)} frames)")


# ==========================================================
# EE trajectory plot (real trajectory)
# ==========================================================
print("\nGenerating EE trajectory plot …")

ee_positions = np.array([p[:3] for p in gt_proprios])  # (T+1, 3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
        "b-o", markersize=3, linewidth=1.5, label="GT EE trajectory")
ax.scatter(*ee_positions[0], c="lime", s=150, marker="^",
           label="Start", zorder=5, edgecolors="k")
ax.scatter(*ee_positions[-1], c="red", s=200, marker="*",
           label="Goal", zorder=5, edgecolors="k")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Ground-Truth EE Trajectory")
ax.legend()
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "ee_trajectory_gt.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path}")


print("\n=== Done ===")
