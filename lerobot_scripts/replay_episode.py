

import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import  SO101Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


from env_cfg import SO101ReplayDatasetEnvCfg


env_cfg = SO101ReplayDatasetEnvCfg()

robot = SO101Follower(env_cfg.robot_cfg)
robot.connect()

dataset = LeRobotDataset(
    repo_id=env_cfg.dataset_name, 
    root=env_cfg.dataset_local_path, 
    episodes=[env_cfg.episode_idx]
)

actions = dataset.select_columns("action")

log_say(f"Replaying episode {env_cfg.episode_idx}")
print(f"Replaying episode {env_cfg.episode_idx}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }
    robot.send_action(action)

    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()