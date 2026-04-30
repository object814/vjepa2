"""
MetaworldVideoDataset - a drop-in replacement for DROIDVideoDataset.

This module mirrors the interface of ``app.vjepa_droid.droid`` so that the VJEPA
predictor training loop can be used *without modification* by simply swapping
the data-loading import:

    # instead of:
    from app.vjepa_droid.droid import init_data
    # use:
    from app.vjepa_droid.metaworld_data import init_data

The dataset reads from the DROID-like layout produced by
``scripts/generate_metaworld_dataset.py``.

Dataset layout (on disk):
    <data_root>/
        episodes.csv              # one absolute path per line
        episode_00000/
            metadata.json
            trajectory.h5
            recordings/MP4/<camera>.mp4
        ...

Single-view mode (multiview=False, default):
    __getitem__ returns the same 5-tuple as DROIDVideoDataset:
        (buffer, actions, states, extrinsics, indices)
    where
        buffer     - (C, T, H, W) float tensor  [after transform]
                     or (T, H, W, 3) uint8 ndarray [without transform]
        actions    - (T-1, 7) float32 ndarray
        states     - (T, 7) float32 ndarray
        extrinsics - (T, 7) float32 ndarray
        indices    - (T,) int64 ndarray       (frame indices within the episode)

Multi-view mode (multiview=True):
    __getitem__ returns a 5-tuple:
        (buffers, actions, states, extrinsics, indices)
    where
        buffers    - list of N (C, T, H, W) float tensors, one per camera view
        actions    - (T-1, 7) float32 ndarray
        states     - (T, 7) float32 ndarray
        extrinsics - (T, 7) float32 ndarray   (from the first camera)
        indices    - (T,) int64 ndarray
"""

import json
import os
from logging import getLogger
from math import ceil

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu

_GLOBAL_SEED = 0
logger = getLogger()


# ─────────────────────────────────────────────────────────────────────────────
# init_data  –  same signature as droid.init_data
# ─────────────────────────────────────────────────────────────────────────────

def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=None,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    multiview=False,
):
    """Create a MetaworldVideoDataset and a DataLoader (same API as droid.init_data).

    Args:
        data_path:        Path to ``episodes.csv``.
        camera_views:     List of camera name strings that exist in the dataset
                          (e.g. ``["topview", "front", "gripperPOV"]``).
                          When ``multiview=False`` (default), one is randomly
                          selected per sample (original behaviour).
                          When ``multiview=True``, *all* cameras are loaded and
                          returned as a list of tensors.
        multiview:        If True, load all camera views per sample instead of
                          randomly picking one.
        (all other args): Identical semantics to ``droid.init_data``.
    """
    if camera_views is None:
        camera_views = ["topview"]

    dataset = MetaworldVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        multiview=multiview,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # Use multiview collator when loading all camera views
    if multiview and collator is None:
        collator = multiview_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("MetaworldVideoDataset data loader created")
    return data_loader, dist_sampler


def multiview_collate(batch):
    """Custom collate for multi-view samples.

    Each sample is (buffers_list, actions, states, extrinsics, indices) where
    buffers_list is a list of N camera tensors each shaped (C, T, H, W).

    Returns:
        clips_list: list of N tensors each (B, C, T, H, W)
        actions:    (B, T-1, 7)
        states:     (B, T, 7)
        extrinsics: (B, T, 7)
        indices:    (B, T)
    """
    buffers_lists, actions, states, extrinsics, indices = zip(*batch)
    num_views = len(buffers_lists[0])
    clips_list = []
    for v in range(num_views):
        clips_list.append(torch.stack([b[v] for b in buffers_lists], dim=0))
    actions = torch.utils.data.default_collate(actions)
    states = torch.utils.data.default_collate(states)
    extrinsics = torch.utils.data.default_collate(extrinsics)
    indices = torch.utils.data.default_collate(indices)
    return clips_list, actions, states, extrinsics, indices


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _get_json(directory):
    """Read the first .json file found in *directory*."""
    for fn in os.listdir(directory):
        if fn.endswith(".json"):
            with open(os.path.join(directory, fn), "r") as f:
                return json.load(f)
    return None


class MetaworldVideoDataset(torch.utils.data.Dataset):
    """Video + state dataset for Metaworld demonstrations.

    Designed to be a drop-in replacement for ``DROIDVideoDataset``.
    """

    def __init__(
        self,
        data_path: str,
        camera_views: list[str] | None = None,
        frameskip: int = 2,
        frames_per_clip: int = 16,
        fps: int | None = 5,
        transform=None,
        multiview: bool = False,
    ):
        """
        Args:
            data_path:       Path to the ``episodes.csv`` index file.
            camera_views:    Camera names available in the dataset.
            frameskip:       Subsample factor applied *after* fps-based sampling
                             (matches DROID's tubelet_size usage).
            frames_per_clip: Number of frames returned per clip.
            fps:             Target frames-per-second to sample from the video.
                             ``None`` means use the video's native fps.
            transform:       Video transform (e.g. ``make_transforms``).
            multiview:       If True, load all camera views per sample instead
                             of randomly picking one.
        """
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.multiview = multiview

        if camera_views is None:
            camera_views = ["topview"]
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = list(
            pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0]
        )
        self.samples = [str(s) for s in samples]

    # ─── __getitem__ ─────────────────────────────────────────────────────

    def __getitem__(self, index):
        path = self.samples[index]

        loaded = False
        while not loaded:
            try:
                buffer, actions, states, extrinsics, indices = self._load_sample(path)
                loaded = True
            except Exception as e:
                logger.info(f"Error loading {path}: {e}")
                index = np.random.randint(len(self))
                path = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    # ─── loading logic ───────────────────────────────────────────────────

    def _load_sample(self, path: str):
        metadata = _get_json(path)
        if metadata is None:
            raise RuntimeError(f"No metadata.json in {path}")

        traj_path = os.path.join(path, self.h5_name)
        traj = h5py.File(traj_path, "r")

        # ── states  (T_full, 7) ──────────────────────────────────────────
        cart_pos = np.array(traj["observation"]["robot_state"]["cartesian_position"])  # (T, 6)
        grip_pos = np.array(traj["observation"]["robot_state"]["gripper_position"])    # (T,)
        states_full = np.concatenate([cart_pos, grip_pos[:, None]], axis=1).astype(np.float32)  # (T, 7)

        # ── determine which cameras to load ──────────────────────────────
        if self.multiview:
            cam_names = self.camera_views  # load all
        else:
            cam_idx = torch.randint(0, len(self.camera_views), (1,)).item()
            cam_names = [self.camera_views[cam_idx]]

        # ── extrinsics  (T_full, 7) — from first camera ─────────────────
        ext_key = cam_names[0]
        if ext_key in traj["observation"]["camera_extrinsics"]:
            extrinsics_full = np.array(
                traj["observation"]["camera_extrinsics"][ext_key]
            ).astype(np.float32)
        else:
            extrinsics_full = np.zeros_like(states_full)

        # ── video: compute shared frame indices from the first camera ────
        first_mp4_rel = metadata.get(cam_names[0])
        if first_mp4_rel is None:
            raise RuntimeError(f"Camera '{cam_names[0]}' not found in metadata at {path}")
        first_vpath = os.path.join(path, first_mp4_rel)
        vr0 = VideoReader(first_vpath, num_threads=-1, ctx=cpu(0))
        vfps = vr0.get_avg_fps()
        fpc = self.frames_per_clip
        target_fps = self.fps if self.fps is not None else vfps
        fstp = max(1, ceil(vfps / target_fps))
        nframes = int(fpc * fstp)
        vlen = len(vr0)

        if vlen < nframes:
            raise RuntimeError(
                f"Video too short: {first_vpath} has {vlen} frames, need {nframes}"
            )

        # random window (shared across all cameras)
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        # ── subsample by frameskip ───────────────────────────────────────
        states = states_full[indices, :][:: self.frameskip]
        extrinsics = extrinsics_full[indices, :][:: self.frameskip]

        # ── actions as state diffs (T-1, 7) ──────────────────────────────
        actions = states[1:] - states[:-1]

        # ── load video frames for each camera ────────────────────────────
        buffers = []
        for cam_name in cam_names:
            mp4_rel = metadata.get(cam_name)
            if mp4_rel is None:
                raise RuntimeError(f"Camera '{cam_name}' not found in metadata at {path}")
            vpath = os.path.join(path, mp4_rel)
            vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
            vr.seek(0)
            buf = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8
            if self.transform is not None:
                buf = self.transform(buf)
            buffers.append(buf)

        traj.close()

        if self.multiview:
            return buffers, actions, states, extrinsics, indices
        else:
            return buffers[0], actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)
