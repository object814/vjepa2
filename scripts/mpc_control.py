"""
MPC Control with V-JEPA2 World Model for Metaworld Pick-Place Task

This script:
1. Loads the V-JEPA2 encoder and trained predictor
2. Initializes two identical Metaworld environments
3. Rolls out one environment using the expert policy
4. Samples 10 intermediate outcomes as targets
5. Uses the MPC controller with the world model to generate actions
6. Executes the actions in the second environment
7. Saves comparison videos of both rollouts

MPC Configuration: 800 samples, 10 CEM iterations, horizon 1
"""

import sys
sys.path.insert(0, "..")

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm

from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader
from utils.mpc_utils import compute_new_pose
from utils.world_model_wrapper import WorldModel

# ---- Metaworld ----
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper
from metaworld.policies import SawyerPickPlaceV3Policy


# ==========================================================
# CONFIG
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MPC Control with V-JEPA2 World Model")
    parser.add_argument("--task-name", type=str, default="pick-place-v3",
                        help="Metaworld task name")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image size for rendering and encoding")
    parser.add_argument("--episode-length", type=int, default=150,
                        help="Maximum episode length")
    parser.add_argument("--num-targets", type=int, default=10,
                        help="Number of intermediate targets to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--encoder-ckpt", type=str, 
                        default="/data/engs-a2i/catz0908/Metaworld/third_party/vjepa2/ckpts/vitg.pt",
                        help="Path to encoder checkpoint")
    parser.add_argument("--predictor-ckpt", type=str,
                        default="/data/engs-a2i/catz0908/Metaworld/third_party/vjepa2/train/metaworld_predictor_run1/latest.pt",
                        help="Path to predictor checkpoint")
    parser.add_argument("--output-dir", type=str, default="./mpc_output",
                        help="Output directory for videos")
    parser.add_argument("--camera-names", nargs="+", default=["topview", "front"],
                        help="Camera names for rendering")
    parser.add_argument("--mpc-samples", type=int, default=800,
                        help="Number of samples for CEM")
    parser.add_argument("--mpc-cem-steps", type=int, default=10,
                        help="Number of CEM iterations")
    parser.add_argument("--mpc-horizon", type=int, default=1,
                        help="MPC rollout horizon")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS for saved videos")
    return parser.parse_args()


# ==========================================================
# Model Loading Utilities
# ==========================================================

def _sanitize_state_dict(state_dict):
    """Remove common prefixes from state dict keys."""
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _find_state_dict(checkpoint, preferred_keys):
    """Find the state dict in a checkpoint."""
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
    """Load a module from a checkpoint file."""
    checkpoint = robust_checkpoint_loader(ckpt_path, map_location=torch.device("cpu"))
    state_dict, loaded_key = _find_state_dict(checkpoint, preferred_keys)
    state_dict = _sanitize_state_dict(state_dict)
    msg = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded {module_name} from {ckpt_path} (key='{loaded_key}') with msg: {msg}")


def load_vjepa2_model(args):
    """Initialize and load V-JEPA2 encoder and predictor."""
    print("Initializing V-JEPA2-AC model...")
    
    encoder, predictor = init_video_model(
        device=args.device,
        patch_size=16,
        max_num_frames=512,
        tubelet_size=2,
        model_name="vit_giant_xformers",
        crop_size=args.image_size,
        pred_depth=24,
        pred_num_heads=16,
        pred_embed_dim=1024,
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
        args.encoder_ckpt,
        preferred_keys=["target_encoder", "encoder", "model"],
        module_name="encoder",
    )
    _load_module_from_ckpt(
        predictor,
        args.predictor_ckpt,
        preferred_keys=["predictor", "model"],
        module_name="predictor",
    )

    encoder = encoder.to(args.device).eval()
    predictor = predictor.to(args.device).eval()
    
    return encoder, predictor


# ==========================================================
# Environment Creation
# ==========================================================

def make_env(task_name, image_size, camera_names, seed=None):
    """Create a Metaworld environment with the multi-image observation wrapper."""
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
    
    if seed is not None:
        env.reset(seed=seed)

    return env


# ==========================================================
# Observation Processing
# ==========================================================

def encode_observation(encoder, transform, obs, camera_names, device, normalize_reps=True):
    """
    Encode the observation image using the V-JEPA2 encoder.
    
    Args:
        encoder: V-JEPA2 encoder
        transform: Image transform pipeline
        obs: Observation dict with 'image' key containing (H, W, 3*N) uint8 array
        camera_names: List of camera names
        device: torch device
        normalize_reps: Whether to apply layer normalization
    
    Returns:
        List of encoded representations, one per camera. Each is (1, tokens, D).
    """
    image = obs["image"]  # (H, W, 3*N)
    H, W, C_total = image.shape
    num_cameras = C_total // 3
    
    encoded_reps = []
    
    for cam_idx in range(num_cameras):
        # Extract single camera view
        start = 3 * cam_idx
        end = 3 * (cam_idx + 1)
        cam_image = image[:, :, start:end]  # (H, W, 3)
        
        # Transform to tensor: (T=1, H, W, C) -> apply transform -> (1, C, T, H, W)
        cam_image = np.expand_dims(cam_image, axis=0)  # (1, H, W, 3)
        clip = transform(cam_image)  # (C, T, H, W)
        clip = clip.unsqueeze(0).to(device)  # (1, C, T, H, W)
        
        # Encode
        B, C, T, H_t, W_t = clip.size()
        # Reshape for encoder: need 2 frames minimum
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        # clip shape: (B*T, C, 2, H, W)
        
        h = encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # (B, T*tokens, D)
        
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        
        encoded_reps.append(h)
    
    return encoded_reps


def proprio_to_state(proprio):
    """
    Convert proprio observation to state format expected by MPC.
    
    Proprio format from wrapper: ee_pos(3) + ee_vel(3) + gripper(1) = 7D
    MPC state format: ee_pos(3) + euler_angles(3) + gripper(1) = 7D
    
    Since Metaworld doesn't provide orientation, we use zeros for euler angles.
    """
    ee_pos = proprio[:3]
    ee_vel = proprio[3:6]
    gripper = proprio[6:7]
    # euler_angles = np.zeros(3, dtype=np.float32)
    
    state = np.concatenate([ee_pos, ee_vel, gripper])
    return state


def action_vjepa_to_metaworld(action_7d):
    """
    Convert V-JEPA action format to Metaworld action format.
    
    V-JEPA action: delta_pos(3) + delta_angles(3) + gripper(1) = 7D
        - delta_pos is an actual position delta in meters (from state diffs during training)
        - gripper is a gripper state value
    
    Metaworld action: normalized_action(3) + gripper(1) = 4D
        - normalized_action is in [-1, 1], internally scaled by action_scale=1/80
        - actual_movement = normalized_action * action_scale
    
    Therefore: normalized_action = delta_pos / action_scale = delta_pos * 80
    """
    delta_pos = action_7d[:3]
    gripper = action_7d[6:7]
    
    # Scale position delta to normalized Metaworld action
    # Metaworld uses action_scale = 1/80 = 0.0125 internally
    # actual_delta = normalized_action * action_scale
    # So: normalized_action = actual_delta / action_scale = actual_delta * 80
    METAWORLD_ACTION_SCALE = 1.0 / 80.0
    normalized_pos = delta_pos / METAWORLD_ACTION_SCALE  # = delta_pos * 80
    
    action_4d = np.concatenate([normalized_pos, gripper])
    return np.clip(action_4d, -1.0, 1.0).astype(np.float32)


# ==========================================================
# Expert Policy Rollout
# ==========================================================

def rollout_expert_policy(env, policy, episode_length):
    """
    Roll out the expert policy in the environment.
    
    Returns:
        dict with 'frames', 'proprios', 'actions', 'original_obs', 'success'
    """
    print("Rolling out expert policy...")
    
    frames = []
    proprios = []
    actions = []
    original_obs_list = []
    
    obs, info = env.reset()
    frames.append(obs["image"].copy())
    proprios.append(obs["proprio"].copy())
    original_obs_list.append(obs["original_obs"].copy())
    
    # Warm-up steps
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    
    frames = [obs["image"].copy()]
    proprios = [obs["proprio"].copy()]
    original_obs_list = [obs["original_obs"].copy()]
    
    for t in tqdm(range(episode_length), desc="Expert rollout"):
        # Get action from expert policy
        action = policy.get_action(obs["original_obs"])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        frames.append(obs["image"].copy())
        proprios.append(obs["proprio"].copy())
        original_obs_list.append(obs["original_obs"].copy())
        actions.append(action.copy())
        
        if terminated or truncated:
            break
    
    success = info.get("success", False)
    print(f"Expert rollout completed. Success: {success}, Steps: {len(actions)}")
    
    return {
        "frames": np.stack(frames, axis=0),  # (T+1, H, W, 3*N)
        "proprios": np.stack(proprios, axis=0),  # (T+1, 7)
        "actions": np.stack(actions, axis=0),  # (T, 4)
        "original_obs": np.stack(original_obs_list, axis=0),  # (T+1, obs_dim)
        "success": success,
    }


# ==========================================================
# MPC Control Loop
# ==========================================================

def run_mpc_control(env, world_model, encoder, transform, expert_data, 
                    num_targets, tokens_per_frame, camera_names, device, episode_length):
    """
    Run MPC control using intermediate targets from expert rollout.
    
    Args:
        env: Metaworld environment
        world_model: WorldModel with MPC
        encoder: V-JEPA2 encoder
        transform: Image transform pipeline
        expert_data: Dict with expert rollout data
        num_targets: Number of intermediate targets
        tokens_per_frame: Number of tokens per frame
        camera_names: List of camera names
        device: torch device
        episode_length: Max steps to run
    
    Returns:
        dict with 'frames', 'proprios', 'actions', 'success'
    """
    print(f"Running MPC control with {num_targets} intermediate targets...")
    
    # Sample target indices evenly from the expert trajectory
    expert_length = len(expert_data["frames"]) - 1  # Exclude initial frame
    target_indices = np.linspace(0, expert_length, num_targets + 1, dtype=int)[1:]  # Skip index 0
    print(f"Target indices: {target_indices}")
    
    # Pre-encode all target frames
    print("Pre-encoding target frames...")
    target_encodings = []
    for target_idx in tqdm(target_indices, desc="Encoding targets"):
        target_obs = {
            "image": expert_data["frames"][target_idx],
            "proprio": expert_data["proprios"][target_idx],
        }
        
        # Use first camera for target encoding
        with torch.no_grad():
            encoded = encode_observation(
                encoder, transform, target_obs, camera_names, device
            )
            # Use first camera's encoding
            target_encodings.append(encoded[0][:, :tokens_per_frame])  # (1, tokens, D)
    
    # Initialize control environment
    obs, info = env.reset()
    
    # Warm-up steps
    for _ in range(5):
        obs, r, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    
    frames = [obs["image"].copy()]
    proprios = [obs["proprio"].copy()]
    actions = []
    
    current_target_idx = 0
    steps_since_target = 0
    max_steps_per_target = episode_length // num_targets + 10  # Allow some extra steps
    
    for step in tqdm(range(episode_length), desc="MPC control"):
        # Get current target encoding
        goal_rep = target_encodings[current_target_idx]
        
        # Encode current observation
        with torch.no_grad():
            current_encodings = encode_observation(
                encoder, transform, obs, camera_names, device
            )
            current_rep = current_encodings[0][:, :tokens_per_frame]  # Use first camera
            
            # Get current state for MPC
            state = proprio_to_state(obs["proprio"])
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Run MPC to get action
            # infer_next_action returns (rollout, 7), so index with [0] for first timestep
            action_7d = world_model.infer_next_action(
                current_rep, 
                state_tensor, 
                goal_rep
            ).cpu().numpy()[0]  # (7,)
        
        # Convert to Metaworld action format
        action_4d = action_vjepa_to_metaworld(action_7d)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_4d)
        
        # Store data
        frames.append(obs["image"].copy())
        proprios.append(obs["proprio"].copy())
        actions.append(action_4d.copy())
        
        steps_since_target += 1
        
        # Check if we should move to next target
        # Either by reaching max steps or by being close to target
        if steps_since_target >= max_steps_per_target:
            if current_target_idx < num_targets - 1:
                current_target_idx += 1
                steps_since_target = 0
                print(f"\nStep {step}: Moving to target {current_target_idx + 1}/{num_targets}")
        
        if terminated or truncated:
            break
    
    success = info.get("success", False)
    print(f"\nMPC control completed. Success: {success}, Steps: {len(actions)}")
    
    return {
        "frames": np.stack(frames, axis=0),
        "proprios": np.stack(proprios, axis=0),
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 4)),
        "success": success,
    }


# ==========================================================
# Video Saving
# ==========================================================

def save_video(frames, output_path, fps=15, camera_idx=0):
    """
    Save frames as a video file.
    
    Args:
        frames: (T, H, W, 3*N) uint8 array
        output_path: Path to save video
        fps: Frames per second
        camera_idx: Which camera view to save
    """
    T, H, W, C_total = frames.shape
    num_cameras = C_total // 3
    
    # Extract single camera view
    start = 3 * camera_idx
    end = 3 * (camera_idx + 1)
    camera_frames = frames[:, :, :, start:end]  # (T, H, W, 3)
    
    # Convert RGB to BGR for OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in camera_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"Saved video: {output_path}")


def save_comparison_video(expert_frames, mpc_frames, output_path, fps=15, camera_idx=0):
    """
    Save side-by-side comparison video of expert and MPC rollouts.
    
    Args:
        expert_frames: (T1, H, W, 3*N) uint8 array
        mpc_frames: (T2, H, W, 3*N) uint8 array
        output_path: Path to save video
        fps: Frames per second
        camera_idx: Which camera view to use
    """
    T1, H, W, C_total = expert_frames.shape
    T2 = mpc_frames.shape[0]
    T = max(T1, T2)
    
    # Extract camera views
    start = 3 * camera_idx
    end = 3 * (camera_idx + 1)
    expert_cam = expert_frames[:, :, :, start:end]
    mpc_cam = mpc_frames[:, :, :, start:end]
    
    # Pad shorter video with last frame
    if T1 < T:
        pad = np.repeat(expert_cam[-1:], T - T1, axis=0)
        expert_cam = np.concatenate([expert_cam, pad], axis=0)
    if T2 < T:
        pad = np.repeat(mpc_cam[-1:], T - T2, axis=0)
        mpc_cam = np.concatenate([mpc_cam, pad], axis=0)
    
    # Create side-by-side frames
    combined_frames = np.concatenate([expert_cam, mpc_cam], axis=2)  # (T, H, 2*W, 3)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled_frames = []
    for i, frame in enumerate(combined_frames):
        frame = frame.copy()
        cv2.putText(frame, "Expert", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MPC", (W + 10, 30), font, 0.7, (255, 255, 255), 2)
        labeled_frames.append(frame)
    
    combined_frames = np.stack(labeled_frames, axis=0)
    
    # Save video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (2 * W, H))
    
    for frame in combined_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"Saved comparison video: {output_path}")


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()
    
    print("=" * 60)
    print("MPC Control with V-JEPA2 World Model")
    print("=" * 60)
    print(f"Task: {args.task_name}")
    print(f"Device: {args.device}")
    print(f"MPC: samples={args.mpc_samples}, CEM_steps={args.mpc_cem_steps}, horizon={args.mpc_horizon}")
    print(f"Targets: {args.num_targets}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load V-JEPA2 model
    encoder, predictor = load_vjepa2_model(args)
    tokens_per_frame = int((args.image_size // encoder.patch_size) ** 2)
    print(f"Tokens per frame: {tokens_per_frame}")
    
    # Create transform
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.image_size,
    )
    
    # Create WorldModel with MPC
    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": args.mpc_horizon,
            "samples": args.mpc_samples,
            "topk": 10,
            "cem_steps": args.mpc_cem_steps,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": False,
        },
        normalize_reps=True,
        device=args.device,
    )
    
    # Create two identical environments
    print("\nCreating environments with seed:", args.seed)
    env_expert = make_env(args.task_name, args.image_size, args.camera_names, seed=args.seed)
    env_mpc = make_env(args.task_name, args.image_size, args.camera_names, seed=args.seed)
    
    # Create expert policy
    policy = SawyerPickPlaceV3Policy()
    
    # Reset both environments with the same seed to get identical initial states
    env_expert.reset(seed=args.seed)
    env_mpc.reset(seed=args.seed)
    
    # Roll out expert policy
    print("\n" + "=" * 60)
    print("PHASE 1: Expert Policy Rollout")
    print("=" * 60)
    expert_data = rollout_expert_policy(env_expert, policy, args.episode_length)
    
    # Run MPC control
    print("\n" + "=" * 60)
    print("PHASE 2: MPC Control")
    print("=" * 60)
    with torch.no_grad():
        mpc_data = run_mpc_control(
            env_mpc,
            world_model,
            encoder,
            transform,
            expert_data,
            args.num_targets,
            tokens_per_frame,
            args.camera_names,
            args.device,
            args.episode_length,
        )
    
    # Save videos
    print("\n" + "=" * 60)
    print("PHASE 3: Saving Videos")
    print("=" * 60)
    
    for cam_idx, cam_name in enumerate(args.camera_names):
        # Save individual videos
        expert_video_path = str(output_dir / f"expert_{args.task_name}_{cam_name}.mp4")
        save_video(expert_data["frames"], expert_video_path, fps=args.fps, camera_idx=cam_idx)
        
        mpc_video_path = str(output_dir / f"mpc_{args.task_name}_{cam_name}.mp4")
        save_video(mpc_data["frames"], mpc_video_path, fps=args.fps, camera_idx=cam_idx)
        
        # Save comparison video
        comparison_path = str(output_dir / f"comparison_{args.task_name}_{cam_name}.mp4")
        save_comparison_video(
            expert_data["frames"], 
            mpc_data["frames"], 
            comparison_path, 
            fps=args.fps, 
            camera_idx=cam_idx
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Expert rollout: {len(expert_data['actions'])} steps, Success: {expert_data['success']}")
    print(f"MPC control: {len(mpc_data['actions'])} steps, Success: {mpc_data['success']}")
    print(f"\nVideos saved to: {output_dir}")
    
    # Cleanup
    env_expert.close()
    env_mpc.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
