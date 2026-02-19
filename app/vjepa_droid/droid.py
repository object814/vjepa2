# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
droid.py
This script defines the DROIDVideoDataset class, which is a PyTorch Dataset for loading video data from the DROID dataset.
It includes functions for initialising the dataset and data loader, as well as processing video frames and associated metadata such as robot states and camera extrinsics.
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
from scipy.spatial.transform import Rotation

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    """
    Initialises the DROIDVideoDataset and creates a DataLoader for it.

    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.
        frames_per_clip (int): Number of frames per video clip.
        fps (int): Frames per second to sample from the videos.
        crop_size (int): Size to crop the video frames to.
        rank (int): Rank of the current process for distributed training.
        world_size (int): Total number of processes for distributed training.
        camera_views (list): List of camera views to use.
        stereo_view (bool): Whether to use stereo camera views.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of worker processes for data loading.
        pin_mem (bool): Whether to pin memory for faster data transfer to GPU.
        persistent_workers (bool): Whether to keep worker processes alive after the initial dataset loading.
        collator (function): Custom collate function for the DataLoader.
        transform (function): Transformations to apply to the video frames.
        camera_frame (bool): Whether to transform states to the camera frame.
        tubelet_size (int): Number of frames to skip between sampled frames (for tubelet sampling).
    Returns:
        DataLoader: A PyTorch DataLoader for the DROIDVideoDataset.
        DistributedSampler: A PyTorch DistributedSampler for the dataset.

    Usage:
        data_loader, dist_sampler = init_data(
            data_path="/path/to/dataset",
            ... # other parameters
        )
        # During training loop:
        for batch in data_loader:
            # process batch
    
    """
    # Create the dataset
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )

    # Create the distributed sampler
    # DistributedSampler will handle shuffling and partitioning of the dataset across multiple processes for distributed training
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # Create the DataLoader
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

    logger.info("VideoDataset unsupervised data loader created")

    return data_loader, dist_sampler


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""
    """
    DROIDVideoDataset is a PyTorch Dataset class designed to load video data from the DROID dataset.
    It reads video files and associated metadata, processes the video frames, and extracts relevant information such as robot states and camera extrinsics.
    """

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        """
        Args:
            data_path (str): Path to the dataset.
            camera_views (list): List of camera views to use (e.g., ["left_mp4_path", "right_mp4_path"]).
            frameskip (int): Number of frames to skip between sampled frames (for tubelet sampling).
            frames_per_clip (int): Number of frames in each clip.
            fps (int): Frames per second to sample from the videos.
            transform (function): Transformations to apply to the video frames.
            camera_frame (bool): Whether to transform states to the camera frame.
        """
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        self.h5_name = "trajectory.h5"

        samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        self.samples = samples

    def __getitem__(self, index):
        """
        Loads a video sample and its associated metadata (actions, states, extrinsics) from the dataset.
        It randomly samples a video clip from the specified camera views, processes the video frames, and extracts the relevant information for training.

        Args:
            index (int): Index of the video sample to load. If the index is invalid, it will keep trying to load videos until it finds a valid sample.

        Returns:
            buffer (numpy array): The video frames for the sampled clip, after applying any specified transformations.
            actions (numpy array): The action differences computed from the robot states for the sampled clip.
            states (numpy array): The robot states for the sampled clip, potentially transformed to the camera frame.
            extrinsics (numpy array): The camera extrinsics for the sampled clip.
            indices (numpy array): The frame indices that were sampled from the video.
        """
        path = self.samples[index]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path)
                loaded_video = True
            except Exception as e:
                logger.info(f"Encountered exception when loading video {path=} {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                path = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    def poses_to_diffs(self, poses):
        """
        Converts poses to differences in position and orientation.

        Args:
            poses (numpy array): The robot poses, shape [T, 7] where T is the number of time steps.
        
        Returns:
            diffs (numpy array): The differences in position and orientation, shape [T-1, 6].
        """
        xyz = poses[:, :3]  # shape [T, 3]
        thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness = poses[:, -1:]
        closedness_delta = closedness[1:] - closedness[:-1]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        """
        Transforms the robot states (rotation + translation) from the world frame to the camera frame using the provided camera extrinsics.

        Args:
            poses (numpy array): The robot poses in the world frame, shape [T, 7] where T is the number of time steps.
            extrinsics (numpy array): The camera extrinsics, shape [T, 7] where T is the number of time steps.
        
        Returns:
            transformed_poses (numpy array): The robot poses transformed to the camera frame, shape [T, 7].
        """
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def loadvideo_decord(self, path):
        """
        Loads a video sample and its associated metadata (actions, states, extrinsics) from the specified path using the Decord library.
        It returns the same information as __getitem__, but is separated out for clarity and to handle potential exceptions during video loading.

        Args:
            path (str): The path to the video sample to load.
        
        Returns:
            buffer (numpy array): The video frames for the sampled clip, after applying any specified transformations.
            actions (numpy array): The action differences computed from the robot states for the sampled clip.
            states (numpy array): The robot states for the sampled clip, potentially transformed to the camera frame.
            extrinsics (numpy array): The camera extrinsics for the sampled clip.
            indices (numpy array): The frame indices that were sampled from the video.
        """
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # -- randomly sample a camera view
        camera_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = np.random.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        actions = self.poses_to_diffs(states)
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)
