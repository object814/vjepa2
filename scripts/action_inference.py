import sys
sys.path.insert(0, "..")

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path

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


def forward_target(c, normalize_reps=True):
    """
    Forward pass through the encoder to get the target representations for the input video frames.
    """
    print("=================================")
    print("Running forward pass through the encoder to get target representations for the input video frames.")
    # DEBUG: Print the shape of the input video frames
    print(f"Input video frames shape: {c.shape}")
    B, C, T, H, W = c.size() # [B, C, T, H, W]
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1) # [B*T, C, 1, H, W] -> [B*T, C, 2, H, W]
    print(f"Input video frames shape after rearranging for encoder: {c.shape}")
    h = encoder(c)
    print(f"Output representations shape from encoder: {h.shape}")
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    print(f"Output representations shape after rearranging: {h.shape}")
    if normalize_reps:
        print("Applying layer normalization to the output representations.")
        h = F.layer_norm(h, (h.size(-1),))
        print(f"Output representations shape after normalization: {h.shape}")
    print("=================================")
    return h


def forward_actions(z, states, nsamples, grid_size=0.075, normalize_reps=True, action_repeat=1):
    """
    Forward pass through the predictor to get the predicted representations for a grid of actions sampled around the initial action in the trajectory.

    Args:
    - z: The input representations from the encoder for the context frames. Shape: [B, N, D]
    - states: The input states for the context frames. Shape: [B, N, S]
    - nsamples: The number of samples to take along each action dimension (total samples = nsamples^3)
    - grid_size: The range of the grid to sample actions from, centered around the initial action in the trajectory. The grid will be sampled from [-grid_size, grid_size] along each action dimension.
    - normalize_reps: Whether to apply layer normalization to the predicted representations from the predictor.
    - action_repeat: The number of times to repeat the sampled actions in the forward pass through the predictor.
    """
    def make_action_grid(grid_size=grid_size):
        action_samples = []
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples += [torch.tensor([da, db, dc, 0, 0, 0, 0], device=z.device, dtype=z.dtype)]
        return torch.stack(action_samples, dim=0).unsqueeze(1)

    print("=================================")
    print("Running forward pass through the predictor to get predicted representations for a grid of sampled actions.")

    # Sample grid of actions
    action_samples = make_action_grid()
    print(f"Sampled grid of actions; num actions = {len(action_samples)}")

    def step_predictor(_z, _a, _s):
        """
        Run the predictor for one step given the current representations, actions and states. 
        Update the robot's pose based on the current pose and action.
        """
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:] # Run the predictor for one step and take the last frame's representation as the output
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:]) # Given current pose and action, compute the new pose using the robot's kinematics
        return _z, _s

    print("Initial full-length context representation shape:", z.shape)
    print("Initial current context representation shape:", z[:, :tokens_per_frame].shape)
    print("Initial current context pose shape:", states[:, :1].shape)

    # Context frame rep and context pose
    z_hat = z[:, :tokens_per_frame].repeat(int(nsamples**3), 1, 1)  # [Sampled actions, Number of tokens, Token dimension]. By repeating the context representation for each sampled action, we can evaluate the effect of each action on the future representation.
    s_hat = states[:, :1].repeat((int(nsamples**3), 1, 1))  # [Sampled actions, 1, 7]. Same here, repeat the current state for each sampled action.
    a_hat = action_samples  # [Sampled actions, 1, 7]

    print("Duplicated current context representations shape:", z_hat.shape)
    print("Duplicated current context pose shape:", s_hat.shape)

    for _ in range(action_repeat):
        # If doing action_repeat, repeat the sampled actions for each step
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    print("=================================")

    return z_hat, s_hat, a_hat

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist()


# ==========================================================
# CONFIG
# ==========================================================

TASK_NAME = "pick-place-v3"
IMAGE_SIZE = 224
T = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NSAMPLES = 5
GRID_SIZE = 0.075
camera_names = ["topview", "front"]

ENCODER_CKPT = "/Metaworld/third_party/vjepa2/ckpts/vitg.pt"
PREDICTOR_CKPT = "/Metaworld/third_party/vjepa2/train/metaworld_predictor_run1/latest.pt"

print("Using device:", DEVICE)


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
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
    print(f"Loaded {module_name} from {ckpt_path} (key='{loaded_key}') with msg: {msg}")


# ==========================================================
# Create Env
# ==========================================================

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
# Load VJEPA2-AC
# ==========================================================

print("Initializing V-JEPA2-AC from training config and loading custom checkpoints...")

encoder, predictor = init_video_model(
    device=DEVICE,
    patch_size=16,
    max_num_frames=512,
    tubelet_size=2,
    model_name="vit_giant_xformers",
    crop_size=IMAGE_SIZE,
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
    ENCODER_CKPT,
    preferred_keys=["target_encoder", "encoder", "model"],
    module_name="encoder",
)
_load_module_from_ckpt(
    predictor,
    PREDICTOR_CKPT,
    preferred_keys=["predictor", "model"],
    module_name="predictor",
)

encoder = encoder.to(DEVICE).eval()
predictor = predictor.to(DEVICE).eval()

tokens_per_frame = int((IMAGE_SIZE // encoder.patch_size) ** 2)

transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=IMAGE_SIZE,
)


# ==========================================================
# Collect rollout
# ==========================================================

env = make_env()
obs, _ = env.reset(seed=0)

frames = []
states = []
actions = []

print("Initialising environment...")
for _ in range(5):
    obs, r, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))

print("Collecting rollout...")

for t in range(T):
    print(f"Simulation Step {t}/{T}")
    action = np.array([0.02, -0.02, 0.01, 0.1])  # Move in a straight line
    obs, r, terminated, truncated, _ = env.step(action)

    frames.append(obs["image"])
    states.append(obs["proprio"])
    actions.append(action)

    if terminated or truncated:
        obs, _ = env.reset()

# Turn observations into tensors with correct shapes
# Frames: list of (H, W, C*number_cameras) -> Three seperate (1, 3, T, H, W) tensors for each camera, where T is the number of frames
# States: list of (state_dim,) -> (1, T, state_dim)
# Actions: list of (action_dim,) -> (1, T, action_dim)
# Convert to numpy arrays
frames = np.stack(frames, axis=0)      # (T, H, W, 3V)
states = np.stack(states, axis=0)      # (T, state_dim)
actions = np.stack(actions, axis=0)    # (T, action_dim)

T, H, W, C_total = frames.shape
num_cameras = C_total // 3

# Split concatenated channels into separate RGB views
camera_frames = []

for cam_idx in range(num_cameras):
    start = 3 * cam_idx
    end = 3 * (cam_idx + 1)

    cam = frames[:, :, :, start:end]  # (T, H, W, 3)
    camera_frames.append(cam)

camera_tensors = []

for cam_idx, cam in enumerate(camera_frames):
    cam = torch.from_numpy(cam).float().to(DEVICE)
    cam = cam.permute(0, 3, 1, 2)
    cam = cam.permute(1, 0, 2, 3)
    cam = cam.unsqueeze(0)
    camera_tensors.append(cam)

# Convert states and actions
states = torch.from_numpy(states).float().unsqueeze(0).to(DEVICE)
actions = torch.from_numpy(actions).float().unsqueeze(0).to(DEVICE)

print("\nFinal tensor shapes:")
for i, cam in enumerate(camera_tensors):
    print(f"Camera {i}: {cam.shape}")
print("States:", states.shape)
print("Actions:", actions.shape)

# ==========================================================
# Forward encoding and Energy Landscape Visualisation
# ==========================================================
print("Running forward encoding and visualising energy landscape for the sampled trajectory...")

nsamples = 5
grid_size = 0.01
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for cam_idx, cam in enumerate(camera_tensors):
        cam_name = camera_names[cam_idx]

        h = forward_target(cam)
        z_hat, s_hat, a_hat = forward_actions(
            h, states, nsamples=nsamples, grid_size=grid_size
        )
        loss = loss_fn(z_hat, h)

        # Build plot data
        plot_data = []
        for b, v in enumerate(loss):
            plot_data.append((
                a_hat[b, :-1, 0].sum(),
                a_hat[b, :-1, 1].sum(),
                a_hat[b, :-1, 2].sum(),
                v,
            ))

        delta_x = [d[0].detach().cpu().item() for d in plot_data]
        delta_z = [d[2].detach().cpu().item() for d in plot_data]
        energy  = [d[3] for d in plot_data]

        heatmap, xedges, yedges = np.histogram2d(
            delta_x, delta_z, weights=energy, bins=nsamples
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis"
        )
        ax.set_xlabel("Action Delta x")
        ax.set_ylabel("Action Delta z")
        ax.set_title(f"Energy Landscape - {cam_name}")
        fig.colorbar(im, ax=ax)

        output_path = os.path.join(
            output_dir, f"energy_heatmap_{cam_name}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved heatmap to {output_path}")

# ==========================================================
# Forward encoder and MPC Planning
# ==========================================================
print("Running Encoder and MPC planning...")
world_model = WorldModel(
    encoder=encoder,
    predictor=predictor,
    tokens_per_frame=tokens_per_frame,
    transform=transform,
    # Doing very few CEM iterations with very few samples just to run efficiently on CPU...
    # ... increase cem_steps and samples for more accurate optimization of energy landscape
    mpc_args={
        "rollout": 2,
        "samples": 25,
        "topk": 10,
        "cem_steps": 2,
        "momentum_mean": 0.15,
        "momentum_mean_gripper": 0.15,
        "momentum_std": 0.75,
        "momentum_std_gripper": 0.15,
        "maxnorm": 0.075,
        "verbose": True
    },
    normalize_reps=True,
    device=DEVICE
)

# We do inference on every camera view separately here
with torch.no_grad():
    for cam_idx, cam in enumerate(camera_tensors):
        h = forward_target(cam)
        z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:]
        s_n = states[:, :1]
        print(f"Starting planning using Cross-Entropy Method for camera {cam_idx}...")
        actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
        print(f"Actions returned by planning with CEM (x,y,z) = ({actions[0, 0]:.2f},{actions[0, 1]:.2f} {actions[0, 2]:.2f})")

env.close()
print("Done.")
