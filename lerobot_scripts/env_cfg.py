
from dataclasses import dataclass
from pathlib import Path


from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode,Cv2Rotation

from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig



@dataclass
class SO101EnvCfg():
    
    robot_port = '/dev/ttyACM0'
    robot_id = "my_follower_arm"
    fps = 30

    height = 480       # 240
    width =  640       # 320

    # camera_cfg = {
    #     "wrist": OpenCVCameraConfig(
    #         index_or_path=0, 
    #         width=width, 
    #         height=height, 
    #         fps=fps, 
    #         color_mode=ColorMode.RGB,
    #         rotation=Cv2Rotation.NO_ROTATION
    #         ),
    #     "top": OpenCVCameraConfig(
    #         index_or_path=2, 
    #         width=width, 
    #         height=height, 
    #         fps=fps, 
    #         color_mode=ColorMode.RGB
    #         )
    # }

    robot_cfg = SO101FollowerConfig(
        port=robot_port, 
        id=robot_id, 
        # cameras=camera_cfg
    )


@dataclass
class DatasetCfg():
    task_description = "Pick the red lego block and put it in the transparent box"  
    dataset_name: str = "pick_and_place_red_lego_v2"
    dataset_local_path: Path | None = None

    def __post_init__(self):
        if self.dataset_local_path is None:
            self.dataset_local_path = Path.home() / "data" / self.dataset_name



@dataclass
class SO101TeleopEnvCfg(SO101EnvCfg):
    teleop_port = '/dev/ttyACM1'
    teleop_id = "my_leader_arm"

    teleop_cfg = SO101LeaderConfig(
        port=teleop_port,
        id=teleop_id,
    )


@dataclass
class SO101TeleopRecordEnvCfg(SO101TeleopEnvCfg, DatasetCfg):
    bool_resume = False
    num_episodes = 50
    episode_time_sec = 90
    reset_time_sec = 6



@dataclass
class SO101ReplayDatasetEnvCfg(SO101EnvCfg, DatasetCfg):
    episode_idx = 3

    


