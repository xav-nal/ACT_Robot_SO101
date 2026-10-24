from pathlib import Path


import torch
from torch.utils.data import DataLoader, random_split

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler

from act_utils import Normalizer
from policy_cfg import PolicyCfg


class DatasetManager:
    def __init__(self, repo_id : str, local_path : Path, episodes : list[int]): 

        # ── 1. Load dataset metadata first (no data download yet) ─────────
        # Load without delta_timestamps to inspect structure
        
        dataset_full = LeRobotDataset(repo_id=repo_id, root=local_path, episodes=episodes)

        self.repo_id = repo_id
        self.local_path = local_path
        self.episodes = episodes

        self.camera_keys = self.get_camera_keys(dataset_full)
        self.fps         = dataset_full.fps

        self.stats = dataset_full.meta.stats
        self.normalizer = Normalizer(self.stats)
        self.total_episodes = dataset_full.meta.total_episodes

        num_cameras = len(self.camera_keys)

        print(f"[INFO] Dataset              : {repo_id}")
        print(f"[INFO] Dataset path         : {local_path}")
        print(f"[INFO] Total cameras        : {num_cameras}")
        print(f"[INFO] Cameras detected     : {self.camera_keys}")
        print(f"[INFO] FPS                  : {self.fps}")
        print(f"[INFO] Total frames         : {dataset_full.meta.total_frames}")
        print(f"[INFO] Total episodes       : {dataset_full.meta.total_episodes}")
        print(f"[INFO] Selected frames      : {dataset_full.num_frames}")
        print(f"[INFO] Selected episodes    : {dataset_full.num_episodes}")



        # Infer dims from a single sample
        sample = dataset_full[0]
        self.state_dim  = sample["observation.state"].shape[-1]
        self.action_dim = sample["action"].shape[-1]

        print(f"[INFO] State dim        : {self.state_dim}")
        print(f"[INFO] Action dim       : {self.action_dim}")



    def get_camera_keys(self, dataset: LeRobotDataset) -> list[str]:
        """
        Auto-detect all camera keys in the dataset.
        LeRobot camera keys always start with 'observation.images.'
        e.g. ['observation.images.top', 'observation.images.wrist']
        """
        return [
            key for key in dataset.features
            if key.startswith("observation.images.")
        ]
    

    def make_delta_timestamps(self, chunk_size: int) -> dict:
        """
        Build delta_timestamps dict for LeRobotDataset so that:
        - each camera returns only the current frame   → [0.0]
        - action returns the next chunk_size frames    → [1/fps, 2/fps, ..., chunk_size/fps]

        delta_timestamps are in seconds relative to the current frame.
        """
        self.chunk_size = chunk_size
        dt = {}

        # # current frame only for each camera
        # for key in self.camera_keys:
        #     dt[key] = [0.0]

        # future action chunk
        dt["action"] = [i / self.fps for i in range(0, chunk_size)]


        return dt
    


    def build_dataloaders(
            self, 
            policy_cfg: PolicyCfg,
            device: torch.device
            ):
        """
        Load a LeRobotDataset, auto-detect cameras/dims, build train/val DataLoaders.
        Returns (train_loader, val_loader, camera_keys, state_dim, action_dim)
        """

        # ── Load Dataset with delta_timestamps for chunk loading ─────────────
        delta_timestamps = self.make_delta_timestamps(chunk_size = policy_cfg.chunk_size)
        # dataset = LeRobotDataset(self.repo_id, delta_timestamps=delta_timestamps)
        # dataset = LeRobotDataset(
        #     repo_id=self.repo_id, 
        #     root=self.local_path,
        #     episodes=self.episodes,
        #     delta_timestamps=delta_timestamps
        #     )

        # ── Train / val split ───────────────────────────────────────────
        # n_val   = max(1, int(len(dataset) * policy_cfg.val_ratio))
        # n_train = len(dataset) - n_val

        episodes = list(range(0, self.total_episodes))
        n_val_eps  = max(1, int(policy_cfg.val_n_episodes))
        n_val_eps = 0

        # train_episodes = episodes[:-n_val_eps]
        # val_episodes   = episodes[-n_val_eps:]

        train_dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.local_path,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=None,
            revision=None,
            video_backend="torchcodec",
            tolerance_s=0.0001,
        )

        print(" ###### delta_timestamps", delta_timestamps)

        # val_dataset = LeRobotDataset(
        #     repo_id=self.repo_id,
        #     root=self.local_path,
        #     episodes=val_episodes,
        #     delta_timestamps=delta_timestamps,
        # )


        # sampler = EpisodeAwareSampler(
        #     train_dataset.meta.episodes["dataset_from_index"],
        #     train_dataset.meta.episodes["dataset_to_index"],
        #     episode_indices_to_use=train_dataset.episodes,
        #     drop_n_last_frames=policy_cfg.drop_n_last_frames,  # from your ACT config
        #     shuffle=True,
        # )


        # generator = torch.Generator().manual_seed(policy_cfg.seed)
        # train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

        train_loader = DataLoader(
            train_dataset,
            batch_size  = policy_cfg.batch_size,
            shuffle     = True,
            sampler = None,
            num_workers = policy_cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if policy_cfg.num_workers > 0 else None,
        )

        
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size  = policy_cfg.batch_size,
        #     shuffle     = True,
        #     num_workers = 0,
        #     pin_memory=device.type == "cuda",
        #     drop_last=False,
        # )


        # return train_loader, val_loader 
        return train_loader


    def process_batch(self, batch: dict, device: torch.device):
        """
        Extract and move tensors from a LeRobot batch to device.

        LeRobot returns images in range [0, 1] as float32 tensors — we apply
        ImageNet normalisation here before passing to ResNet.
        

        Returns:
            images      : (B, num_cameras, 3, H, W)  normalised with for example h=240, w=320
            robot_state : (B, state_dim)
            actions     : (B, chunk_size, action_dim)
        """

        # ── Images ────────────────────────────────────────────────────────
        # Each camera key gives (B, 1, C, H, W) because delta=[0.0] → T=1
        # We squeeze T and stack cameras
        cam_images = []
        for key in self.camera_keys:
            img = batch[key].to(device).squeeze(1)           # (B, C, H, W) 
            img = self.normalizer.normalize_image(img, key)       # normalise using dataset stats
            cam_images.append(img)


        # Stack into (B, num_cameras, C, H, W)
        images = torch.stack(cam_images, dim=1)  # (B, num_cameras, 3, H, W)

        # ── State ──────────────────────────────────────────────────────────
        robot_state = batch["observation.state"].to(device)           # (B, state_dim)
        robot_state = self.normalizer.normalize_state(robot_state)         # normalize


        # ── Actions ────────────────────────────────────────────────────────
        # delta_timestamps gives (B, chunk_size, action_dim)
        actions = batch["action"].to(device)                        # (B, chunk_size, action_dim)
        actions = self.normalizer.normalize_action(actions)              # normalize
        action_is_pad = batch["action_is_pad"].to(device)           # (B, chunk_size) False = real, True = pad


        return images, robot_state, actions, action_is_pad
    

    def get_stats(self):
        return self.stats
    



