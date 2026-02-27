"""
Energy Landscape Sweep (Ground-Truth Action)
===========================================
Rolls out a ground-truth trajectory in MetaWorld, then at each timestep
sweeps a scalar multiplier around the ground-truth action:

    a(alpha) = alpha * a_gt

For each alpha, the V-JEPA2 predictor produces a one-step latent prediction
from the current frame/state, and we compute L1 distance to the goal latent
(encoded from the final real frame of the rollout).

This script helps check whether the learned energy landscape is smooth around
and towards the goal-driving action.

Outputs (saved to ./output_energy_landscape/):
  - Per-camera heatmap: energy vs timestep and action scale alpha
  - Per-camera mean±std energy-vs-alpha curve
  - Per-camera best-alpha-over-time plot
    - (diagnostic=split/all) Separate xyz-scale and gripper-scale curves
    - (diagnostic=grid2d/all) Joint xyz-vs-gripper scale 2-D heatmap
"""

import sys
sys.path.insert(0, "..")

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper


def _sanitize_state_dict(state_dict):
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


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

    raise KeyError(f"No state_dict found. Available keys: {list(checkpoint.keys())}")


def _resolve_ckpt_path(path_str):
    path = Path(path_str)
    if path.exists():
        return str(path)

    if path_str.startswith("/Metaworld/"):
        remapped = BASE_DIR / path_str.removeprefix("/Metaworld/")
        if remapped.exists():
            return str(remapped)

    return path_str


def _load_module_from_ckpt(module, ckpt_path, preferred_keys, module_name):
    resolved_path = _resolve_ckpt_path(ckpt_path)
    checkpoint = robust_checkpoint_loader(resolved_path, map_location=torch.device("cpu"))
    state_dict, loaded_key = _find_state_dict(checkpoint, preferred_keys)
    state_dict = _sanitize_state_dict(state_dict)
    msg = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded {module_name} from {resolved_path} (key='{loaded_key}') -> {msg}")


def make_env(task_name, image_size, camera_names):
    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        render_mode="rgb_array",
        max_episode_steps=250,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=image_size,
        image_width=image_size,
        camera_names=camera_names,
    )
    return env


def encode_single_frame(image_np, cam_idx, encoder, device):
    cam = image_np[:, :, 3 * cam_idx: 3 * (cam_idx + 1)]
    tensor = torch.from_numpy(cam).float().to(device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(1).unsqueeze(0)  # (1,3,1,H,W)

    batch_size, channels, num_frames, height, width = tensor.size()
    tensor = tensor.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)

    reps = encoder(tensor)
    reps = reps.view(batch_size, num_frames, -1, reps.size(-1)).flatten(1, 2)
    reps = F.layer_norm(reps, (reps.size(-1),))
    return reps


def evaluate_alpha_sweep(z_current, s_current, z_goal, gt_action_4d,
                         predictor, tokens_per_frame, alphas, device):
    energies = []
    for alpha in alphas:
        action_7d = np.array([
            alpha * gt_action_4d[0],
            alpha * gt_action_4d[1],
            alpha * gt_action_4d[2],
            0.0,
            0.0,
            0.0,
            alpha * gt_action_4d[3],
        ], dtype=np.float32)

        action_tensor = torch.from_numpy(action_7d).unsqueeze(0).unsqueeze(0).to(device)
        z_pred = predictor(z_current, action_tensor, s_current)[:, -tokens_per_frame:]
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

        energy = torch.mean(torch.abs(z_pred - z_goal)).item()
        energies.append(energy)

    return np.asarray(energies, dtype=np.float32)


def evaluate_component_sweep(z_current, s_current, z_goal, gt_action_4d,
                             predictor, tokens_per_frame,
                             xyz_alphas, grip_alphas, device):
    xyz_energies = []
    for alpha_xyz in xyz_alphas:
        action_7d = np.array([
            alpha_xyz * gt_action_4d[0],
            alpha_xyz * gt_action_4d[1],
            alpha_xyz * gt_action_4d[2],
            0.0,
            0.0,
            0.0,
            gt_action_4d[3],
        ], dtype=np.float32)
        action_tensor = torch.from_numpy(action_7d).unsqueeze(0).unsqueeze(0).to(device)
        z_pred = predictor(z_current, action_tensor, s_current)[:, -tokens_per_frame:]
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
        energy = torch.mean(torch.abs(z_pred - z_goal)).item()
        xyz_energies.append(energy)

    grip_energies = []
    for alpha_grip in grip_alphas:
        action_7d = np.array([
            gt_action_4d[0],
            gt_action_4d[1],
            gt_action_4d[2],
            0.0,
            0.0,
            0.0,
            alpha_grip * gt_action_4d[3],
        ], dtype=np.float32)
        action_tensor = torch.from_numpy(action_7d).unsqueeze(0).unsqueeze(0).to(device)
        z_pred = predictor(z_current, action_tensor, s_current)[:, -tokens_per_frame:]
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
        energy = torch.mean(torch.abs(z_pred - z_goal)).item()
        grip_energies.append(energy)

    return (
        np.asarray(xyz_energies, dtype=np.float32),
        np.asarray(grip_energies, dtype=np.float32),
    )


def evaluate_joint_2d_grid(z_current, s_current, z_goal, gt_action_4d,
                           predictor, tokens_per_frame,
                           xyz_alphas, grip_alphas, device):
    energy_grid = np.zeros((len(xyz_alphas), len(grip_alphas)), dtype=np.float32)

    for i, alpha_xyz in enumerate(xyz_alphas):
        for j, alpha_grip in enumerate(grip_alphas):
            action_7d = np.array([
                alpha_xyz * gt_action_4d[0],
                alpha_xyz * gt_action_4d[1],
                alpha_xyz * gt_action_4d[2],
                0.0,
                0.0,
                0.0,
                alpha_grip * gt_action_4d[3],
            ], dtype=np.float32)
            action_tensor = torch.from_numpy(action_7d).unsqueeze(0).unsqueeze(0).to(device)
            z_pred = predictor(z_current, action_tensor, s_current)[:, -tokens_per_frame:]
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
            energy_grid[i, j] = torch.mean(torch.abs(z_pred - z_goal)).item()

    return energy_grid


def save_heatmap(energy_matrix, alphas, cam_name, output_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        energy_matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[alphas[0], alphas[-1], 0, energy_matrix.shape[0] - 1],
    )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="GT scale (alpha=1)")
    ax.set_xlabel("Action scale alpha")
    ax.set_ylabel("Timestep")
    ax.set_title(f"Energy Landscape (L1 to goal) - {cam_name}")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="L1 energy")
    plt.tight_layout()

    path = os.path.join(output_dir, f"energy_heatmap_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_alpha_curve(energy_matrix, alphas, cam_name, output_dir):
    mean_energy = energy_matrix.mean(axis=0)
    std_energy = energy_matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(alphas, mean_energy, "b-", linewidth=2.0, label="Mean energy")
    ax.fill_between(
        alphas,
        mean_energy - std_energy,
        mean_energy + std_energy,
        color="blue",
        alpha=0.2,
        label="±1 std",
    )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="GT scale (alpha=1)")

    best_idx = int(mean_energy.argmin())
    ax.scatter(alphas[best_idx], mean_energy[best_idx], c="cyan", s=70,
               edgecolors="k", zorder=3, label=f"Min mean @ alpha={alphas[best_idx]:.2f}")

    ax.set_xlabel("Action scale alpha")
    ax.set_ylabel("L1 energy")
    ax.set_title(f"Mean Energy vs Action Scale - {cam_name}")
    ax.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(output_dir, f"energy_curve_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, float(alphas[best_idx]), float(mean_energy[best_idx])


def save_best_alpha_plot(energy_matrix, alphas, cam_name, output_dir):
    best_indices = energy_matrix.argmin(axis=1)
    best_alphas = alphas[best_indices]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(best_alphas, "o-", color="purple", markersize=3, linewidth=1.2,
            label="Best alpha per timestep")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="GT scale (alpha=1)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Best alpha")
    ax.set_title(f"Best Action Scale Over Time - {cam_name}")
    ax.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(output_dir, f"best_alpha_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, float(best_alphas.mean())


def save_split_curve_plot(matrix_xyz, matrix_grip,
                          alphas_xyz, alphas_grip, cam_name, output_dir):
    mean_xyz = matrix_xyz.mean(axis=0)
    std_xyz = matrix_xyz.std(axis=0)
    mean_grip = matrix_grip.mean(axis=0)
    std_grip = matrix_grip.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(alphas_xyz, mean_xyz, "b-", linewidth=2, label="Mean energy")
    ax1.fill_between(alphas_xyz, mean_xyz - std_xyz, mean_xyz + std_xyz,
                     color="blue", alpha=0.2, label="±1 std")
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=1.5,
                label="GT xyz scale (1.0)")
    best_xyz_idx = int(mean_xyz.argmin())
    ax1.scatter(alphas_xyz[best_xyz_idx], mean_xyz[best_xyz_idx], c="cyan", s=65,
                edgecolors="k", zorder=3, label=f"Min @ {alphas_xyz[best_xyz_idx]:.2f}")
    ax1.set_title(f"XYZ-scale diagnostic - {cam_name}")
    ax1.set_xlabel("XYZ scale alpha")
    ax1.set_ylabel("L1 energy")
    ax1.legend(loc="best")

    ax2.plot(alphas_grip, mean_grip, "g-", linewidth=2, label="Mean energy")
    ax2.fill_between(alphas_grip, mean_grip - std_grip, mean_grip + std_grip,
                     color="green", alpha=0.2, label="±1 std")
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1.5,
                label="GT grip scale (1.0)")
    best_grip_idx = int(mean_grip.argmin())
    ax2.scatter(alphas_grip[best_grip_idx], mean_grip[best_grip_idx], c="cyan", s=65,
                edgecolors="k", zorder=3, label=f"Min @ {alphas_grip[best_grip_idx]:.2f}")
    ax2.set_title(f"Gripper-scale diagnostic - {cam_name}")
    ax2.set_xlabel("Gripper scale alpha")
    ax2.set_ylabel("L1 energy")
    ax2.legend(loc="best")

    plt.tight_layout()
    path = os.path.join(output_dir, f"diagnostic_split_curve_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return (
        path,
        float(alphas_xyz[best_xyz_idx]),
        float(mean_xyz[best_xyz_idx]),
        float(alphas_grip[best_grip_idx]),
        float(mean_grip[best_grip_idx]),
    )


def save_joint_2d_heatmap(mean_grid, alphas_xyz, alphas_grip, cam_name, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        mean_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[alphas_grip[0], alphas_grip[-1], alphas_xyz[0], alphas_xyz[-1]],
    )
    gt_grip, gt_xyz = 1.0, 1.0
    best_idx = np.unravel_index(int(mean_grid.argmin()), mean_grid.shape)
    best_xyz = alphas_xyz[best_idx[0]]
    best_grip = alphas_grip[best_idx[1]]

    ax.plot(gt_grip, gt_xyz, "*", color="red", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8, label="GT (1,1)")
    ax.plot(best_grip, best_xyz, "o", color="cyan", markersize=9,
            markeredgecolor="k", markeredgewidth=0.8,
            label=f"Min ({best_xyz:.2f}, {best_grip:.2f})")
    ax.set_xlabel("Gripper scale alpha")
    ax.set_ylabel("XYZ scale alpha")
    ax.set_title(f"Joint XYZ/Grip Energy Diagnostic - {cam_name}")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax, label="Mean L1 energy")
    plt.tight_layout()

    path = os.path.join(output_dir, f"diagnostic_joint2d_{cam_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, float(best_xyz), float(best_grip), float(mean_grid[best_idx])


def main():
    parser = argparse.ArgumentParser(description="V-JEPA2 energy landscape around GT action")
    parser.add_argument("--model", type=str, default="giant", choices=["giant", "large"])
    parser.add_argument("--task", type=str, default="compo-draweropen-pickplace")
    parser.add_argument("--steps", type=int, default=40, help="GT rollout length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-alpha", type=int, default=41, help="Number of alpha sweep points")
    parser.add_argument("--alpha-min", type=float, default=-0.5)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default="./output_energy_landscape")
    parser.add_argument(
        "--diagnostic-mode",
        type=str,
        default="all",
        choices=["none", "split", "grid2d", "all"],
        help="Additional diagnostics: split xyz/gripper sweeps, joint 2D sweep, or both",
    )
    parser.add_argument(
        "--diagnostic-max-steps",
        type=int,
        default=25,
        help="Max rollout steps used for extra diagnostics to control runtime",
    )
    parser.add_argument(
        "--diag-num-alpha-xyz",
        type=int,
        default=31,
        help="Resolution of xyz alpha sweep for diagnostics",
    )
    parser.add_argument(
        "--diag-num-alpha-grip",
        type=int,
        default=31,
        help="Resolution of gripper alpha sweep for diagnostics",
    )
    parser.add_argument(
        "--gt-action",
        type=float,
        nargs=4,
        default=[0.2, -0.2, 0.1, 0.1],
        metavar=("DX", "DY", "DZ", "GRIP"),
        help="Ground-truth 4D action to replay (dx dy dz gripper)",
    )
    args = parser.parse_args()

    model_configs = {
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
            "predictor_ckpt": "/Metaworld/third_party/vjepa2/train/metaworld_pickplace_vitl_0224/e275.pt",
            "pred_depth": 12,
            "pred_num_heads": 12,
            "pred_embed_dim": 384,
        },
    }

    cfg = model_configs[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera_names = ["topview", "front"]
    gt_action = np.asarray(args.gt_action, dtype=np.float32)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alpha).astype(np.float32)
    diag_xyz_alphas = np.linspace(args.alpha_min, args.alpha_max, args.diag_num_alpha_xyz).astype(np.float32)
    diag_grip_alphas = np.linspace(args.alpha_min, args.alpha_max, args.diag_num_alpha_grip).astype(np.float32)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using model: {args.model} ({cfg['model_name']})")
    print(f"Using device: {device}")
    print(f"Task: {args.task}")
    print(f"GT action: {gt_action.tolist()}")
    print(f"Alpha sweep: [{args.alpha_min}, {args.alpha_max}] with {args.num_alpha} points")
    print(f"Diagnostic mode: {args.diagnostic_mode}")

    print("\nInitializing V-JEPA2 ...")
    encoder, predictor = init_video_model(
        device=device,
        patch_size=16,
        max_num_frames=512,
        tubelet_size=2,
        model_name=cfg["model_name"],
        crop_size=args.image_size,
        pred_depth=cfg["pred_depth"],
        pred_num_heads=cfg["pred_num_heads"],
        pred_embed_dim=cfg["pred_embed_dim"],
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
        encoder,
        cfg["encoder_ckpt"],
        preferred_keys=["target_encoder", "encoder", "model"],
        module_name="encoder",
    )
    _load_module_from_ckpt(
        predictor,
        cfg["predictor_ckpt"],
        preferred_keys=["predictor", "model"],
        module_name="predictor",
    )

    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    tokens_per_frame = int((args.image_size // encoder.patch_size) ** 2)

    print("\nCollecting ground-truth rollout ...")
    env = make_env(args.task, args.image_size, camera_names)
    obs, _ = env.reset(seed=args.seed)

    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    gt_images = [obs["image"].copy()]
    gt_proprios = [obs["proprio"].copy()]
    gt_actions = []

    for _ in range(args.steps):
        obs, _, terminated, truncated, _ = env.step(gt_action)
        gt_images.append(obs["image"].copy())
        gt_proprios.append(obs["proprio"].copy())
        gt_actions.append(gt_action.copy())
        if terminated or truncated:
            break

    env.close()
    actual_steps = len(gt_actions)
    print(f"Collected {actual_steps} GT steps ({actual_steps + 1} frames).")

    if actual_steps == 0:
        raise RuntimeError("No rollout steps collected. Check task and reset/action configuration.")

    print("\nEncoding goal frame ...")
    camera_z_goals = {}
    with torch.no_grad():
        for cam_idx, cam_name in enumerate(camera_names):
            camera_z_goals[cam_idx] = encode_single_frame(gt_images[-1], cam_idx, encoder, device)
            print(f"  {cam_name}: goal rep shape = {tuple(camera_z_goals[cam_idx].shape)}")

    print("\nEvaluating alpha sweeps ...")
    energy_by_cam = {cam_idx: [] for cam_idx in range(len(camera_names))}

    with torch.no_grad():
        for timestep in range(actual_steps):
            image_t = gt_images[timestep]
            proprio_t = gt_proprios[timestep]
            gt_action_t = gt_actions[timestep]

            s_t = torch.from_numpy(proprio_t).float().unsqueeze(0).unsqueeze(0).to(device)

            for cam_idx, cam_name in enumerate(camera_names):
                z_t = encode_single_frame(image_t, cam_idx, encoder, device)
                z_goal = camera_z_goals[cam_idx]

                energies = evaluate_alpha_sweep(
                    z_t, s_t, z_goal, gt_action_t,
                    predictor, tokens_per_frame, alphas, device,
                )
                energy_by_cam[cam_idx].append(energies)

                if timestep % 5 == 0 or timestep == actual_steps - 1:
                    min_idx = int(energies.argmin())
                    print(
                        f"  t={timestep:03d} | {cam_name:>8s} | "
                        f"best alpha={alphas[min_idx]:.3f} "
                        f"energy={energies[min_idx]:.4f} "
                        f"(alpha=1 energy={energies[np.argmin(np.abs(alphas - 1.0))]:.4f})"
                    )

    split_enabled = args.diagnostic_mode in ("split", "all")
    grid2d_enabled = args.diagnostic_mode in ("grid2d", "all")

    diag_steps = min(actual_steps, args.diagnostic_max_steps)
    split_xyz_by_cam = {cam_idx: [] for cam_idx in range(len(camera_names))}
    split_grip_by_cam = {cam_idx: [] for cam_idx in range(len(camera_names))}
    grid2d_by_cam = {cam_idx: [] for cam_idx in range(len(camera_names))}

    if split_enabled or grid2d_enabled:
        print(f"\nRunning extra diagnostics on first {diag_steps} steps ...")
        with torch.no_grad():
            for timestep in range(diag_steps):
                image_t = gt_images[timestep]
                proprio_t = gt_proprios[timestep]
                gt_action_t = gt_actions[timestep]
                s_t = torch.from_numpy(proprio_t).float().unsqueeze(0).unsqueeze(0).to(device)

                for cam_idx, cam_name in enumerate(camera_names):
                    z_t = encode_single_frame(image_t, cam_idx, encoder, device)
                    z_goal = camera_z_goals[cam_idx]

                    if split_enabled:
                        xyz_energies, grip_energies = evaluate_component_sweep(
                            z_t, s_t, z_goal, gt_action_t,
                            predictor, tokens_per_frame,
                            diag_xyz_alphas, diag_grip_alphas, device,
                        )
                        split_xyz_by_cam[cam_idx].append(xyz_energies)
                        split_grip_by_cam[cam_idx].append(grip_energies)

                    if grid2d_enabled:
                        grid2d = evaluate_joint_2d_grid(
                            z_t, s_t, z_goal, gt_action_t,
                            predictor, tokens_per_frame,
                            diag_xyz_alphas, diag_grip_alphas, device,
                        )
                        grid2d_by_cam[cam_idx].append(grid2d)

                if timestep % 5 == 0 or timestep == diag_steps - 1:
                    print(f"  diagnostic step {timestep:03d}/{diag_steps - 1:03d}")

    print("\nSaving plots ...")
    for cam_idx, cam_name in enumerate(camera_names):
        energy_matrix = np.stack(energy_by_cam[cam_idx], axis=0)

        heatmap_path = save_heatmap(energy_matrix, alphas, cam_name, args.output_dir)
        curve_path, min_alpha, min_energy = save_alpha_curve(
            energy_matrix, alphas, cam_name, args.output_dir
        )
        best_alpha_path, mean_best_alpha = save_best_alpha_plot(
            energy_matrix, alphas, cam_name, args.output_dir
        )

        print(f"  Saved {heatmap_path}")
        print(f"  Saved {curve_path}")
        print(f"  Saved {best_alpha_path}")
        print(
            f"  {cam_name}: min mean energy at alpha={min_alpha:.3f} "
            f"(energy={min_energy:.4f}), mean best alpha over time={mean_best_alpha:.3f}"
        )

        if split_enabled and split_xyz_by_cam[cam_idx] and split_grip_by_cam[cam_idx]:
            split_xyz_matrix = np.stack(split_xyz_by_cam[cam_idx], axis=0)
            split_grip_matrix = np.stack(split_grip_by_cam[cam_idx], axis=0)
            (
                split_path,
                best_xyz_alpha,
                best_xyz_energy,
                best_grip_alpha,
                best_grip_energy,
            ) = save_split_curve_plot(
                split_xyz_matrix,
                split_grip_matrix,
                diag_xyz_alphas,
                diag_grip_alphas,
                cam_name,
                args.output_dir,
            )
            print(f"  Saved {split_path}")
            print(
                f"  {cam_name} split diagnostic: best xyz alpha={best_xyz_alpha:.3f} "
                f"(energy={best_xyz_energy:.4f}), best grip alpha={best_grip_alpha:.3f} "
                f"(energy={best_grip_energy:.4f})"
            )

        if grid2d_enabled and grid2d_by_cam[cam_idx]:
            mean_grid = np.stack(grid2d_by_cam[cam_idx], axis=0).mean(axis=0)
            joint2d_path, joint_best_xyz, joint_best_grip, joint_best_e = save_joint_2d_heatmap(
                mean_grid,
                diag_xyz_alphas,
                diag_grip_alphas,
                cam_name,
                args.output_dir,
            )
            print(f"  Saved {joint2d_path}")
            print(
                f"  {cam_name} joint2d diagnostic: best (xyz alpha, grip alpha)="
                f"({joint_best_xyz:.3f}, {joint_best_grip:.3f}), energy={joint_best_e:.4f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
