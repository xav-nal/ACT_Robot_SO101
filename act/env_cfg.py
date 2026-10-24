
from dataclasses import dataclass, field
from pathlib import Path


from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode,Cv2Rotation

from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig



@dataclass
class SO101EnvCfg():
    
    robot_port : str = '/dev/ttyACM0'
    robot_id : str = "my_follower_arm"
    fps: int = 30


    height = 480       # 240
    width =  640       # 320

    camera_cfg = {
        "wrist.left": OpenCVCameraConfig(index_or_path=0, width=width, height=height, fps=fps, color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION),
        "top": OpenCVCameraConfig(index_or_path=2, width=width, height=height, fps=fps, color_mode=ColorMode.RGB)
    }


    robot_cfg = SO101FollowerConfig(
        port=robot_port, 
        id=robot_id, 
        cameras=camera_cfg
    )


@dataclass
class DatasetCfg():
    task_description = "Pick the red lego block and put it in the box"  
    dataset_name: str = "pick_and_place_red_lego"  # "picking_cube_dataset_v1"
    dataset_episodes: list[int] | None = field(default=None) # field(default_factory=lambda: list(range(0, 40)))
    dataset_local_path: str | None = None
    fps: int = 30


    def __post_init__(self):
        if self.dataset_local_path is None:
            self.dataset_local_path = Path.home() / "data" / "pick_and_place_red_lego"



@dataclass
class SO101ImitationLearningEnvCfg(DatasetCfg):

    device : str = "cuda"
    
   




@dataclass
class SO101ImitationLearningPlayEnvCfg(SO101EnvCfg, DatasetCfg):
    device : str = "cpu"


    max_step_per_episode = 2700

    camera_keys = ['wrist.left', 'top'] # ['front', 'up']


    joint_names = ['shoulder_pan.pos',
        'shoulder_lift.pos',
        'elbow_flex.pos',
        'wrist_flex.pos',
        'wrist_roll.pos',
        'gripper.pos']

