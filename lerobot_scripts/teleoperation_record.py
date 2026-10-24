
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.robots.so_follower import SO101Follower

from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors



from env_cfg import SO101TeleopRecordEnvCfg
from evedev_listener import init_evdev_keyboard_listener


env_cfg = SO101TeleopRecordEnvCfg()

robot = SO101Follower(env_cfg.robot_cfg)
teleop = SO101Leader(env_cfg.teleop_cfg)


action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}




if env_cfg.bool_resume:
    print("Resume recording for dataset: ",env_cfg.dataset_name)
    dataset = LeRobotDataset.resume(
        repo_id=env_cfg.dataset_name,
        root=env_cfg.dataset_local_path,

        image_writer_threads=4,
    )
else:
    print("Create recording for dataset: ",env_cfg.dataset_name)
    dataset = LeRobotDataset.create(
        repo_id=env_cfg.dataset_name,
        root=env_cfg.dataset_local_path,
        fps=env_cfg.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

_, events = init_evdev_keyboard_listener("/dev/input/event3")


init_rerun(session_name="recording")

robot.connect()
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < env_cfg.num_episodes and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {env_cfg.num_episodes}")
    print(f"Recording episode {episode_idx + 1} of {env_cfg.num_episodes}")
    record_loop(
        robot=robot,
        events=events,
        fps=env_cfg.fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=env_cfg.episode_time_sec,
        single_task=env_cfg.task_description,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < env_cfg.num_episodes - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        print("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=env_cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=env_cfg.reset_time_sec,
            single_task=env_cfg.task_description,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        print("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.finalize()