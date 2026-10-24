from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode,Cv2Rotation

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

from lerobot.utils.control_utils import init_keyboard_listener

from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.visualization_utils import log_rerun_data


from env_cfg import SO101TeleopEnvCfg
from evedev_listener import init_evdev_keyboard_listener


env_cfg = SO101TeleopEnvCfg()

robot = SO101Follower(env_cfg.robot_cfg)
teleop = SO101Leader(env_cfg.teleop_cfg)


_, events = init_evdev_keyboard_listener("/dev/input/event7")

init_rerun(session_name="viz")


robot.connect()
teleop.connect()


print("Start the loop")
while True and not events["exit_early"]:
    observation = robot.get_observation()

    action = teleop.get_action()
    robot.send_action(action)

    log_rerun_data(observation=observation, action=action)


robot.disconnect()
teleop.disconnect()

