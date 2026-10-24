

import os
import torch

from lerobot.robots.so_follower import SO101Follower

from lerobot.types import PolicyAction, RobotAction


from act_policy import ACT 
from policy_cfg import  ActCfg
from act_utils import Normalizer



from env_cfg import SO101ImitationLearningPlayEnvCfg




def custom_make_robot_action(action_tensor: PolicyAction, action_names: list[str]) -> RobotAction:
    """Converts a policy's output tensor into a dictionary of named actions.

    This function translates the numerical output from a policy model into a
    human-readable and robot-consumable format, where each dimension of the
    action tensor is mapped to a named motor or actuator command.

    Args:
        action_tensor: e.g., shape [1, action_dim]).
        ds_features: A configuration dictionary containing metadata, including
            the names corresponding to each index of the action tensor.

    Returns:
        A dictionary mapping action names (e.g., "joint_1_motor") to their
        corresponding floating-point values, ready to be sent to a robot
        controller.
    """

    # print("action_tensor", action_tensor.shape)
    
    action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")

    # print("action_tensor", action_tensor.shape)


    act_processed_policy: RobotAction = {
        f"{name}": float(action_tensor[i]) for i, name in enumerate(action_names)
    }
    return act_processed_policy



def process_observation(
        raw_obs : dict, 
        joint_names: list[str], 
        camera_keys: list[str], 
        normalizer: Normalizer, 
        device: torch.device
    ):
    """
    Extract and move tensors from a LeRobot batch to device.

    LeRobot returns images in range [0, 1] as float32 tensors — we apply
    ImageNet normalisation here before passing to ResNet.

    Returns:
        images      : (B, num_cameras, 3, H, W)  normalised
        robot_state : (B, state_dim)
        actions     : (B, chunk_size, action_dim)
    """

    # ── Extract Obs────────────────────────────────────────────────────
    values = [raw_obs[name] for name in joint_names]  # extract in order
    raw_state = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0) # (1, state_dim)
    # print("raw_state", raw_state.shape)
    # print("raw_state", raw_state)


    # ── Images ────────────────────────────────────────────────────────
    # Each camera key gives (B, 1, C, H, W) because delta=[0.0] → T=1
    # We squeeze T and stack cameras
    cam_images = []
    for key in camera_keys:
        # print("key",key)
        # print("raw_obs[key]",raw_obs[key].shape)
        # print("raw_obs[key]",raw_obs[key])
        raw_img = torch.from_numpy(raw_obs[key].transpose(2, 0, 1)).float().div_(255.0).to(device)   # (h=240, w=320, 3) ->  (3, h=240, w=320)
        
        # print("raw_images shape", raw_img.shape)
        img = raw_img.unsqueeze(0)           # (3, h=240, w=320) -> (1, W, H, C)  (h=240, w=320, 3) 
        # print("img squeeze", img.shape) 
        # print("img type", img.dtype)
        img = normalizer.normalize_image(img, key)       # normalise using dataset stats
        cam_images.append(img)


    # Stack into (B, num_cameras, C, H, W)
    images = torch.stack(cam_images, dim=1)  # (1, num_cameras, 3, H, W)

    # ── State ──────────────────────────────────────────────────────────
    robot_state = normalizer.normalize_state(raw_state)            # normalize


    return images, robot_state


def play(env_cfg : SO101ImitationLearningPlayEnvCfg, policy_cfg : ActCfg, device : torch.device):

    path_to_checkpoints = os.path.join(env_cfg.log_dir, "checkpoints", policy_cfg.checkpoint_to_load)
    print("path to checkpoints", path_to_checkpoints)

    ckpt = torch.load(path_to_checkpoints, map_location="cpu", weights_only=False)

    # act_cfg = ActCfg(**ckpt["config"])
    model = ACT(policy_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = Normalizer(ckpt["dataset_stats"])

    # model = ACT.from_pretrained(path_to_checkpoints, "cpu")


    robot = SO101Follower(env_cfg.robot_cfg)
    robot.connect()

    for _ in range(env_cfg.max_step_per_episode):

        # raw_obs : dict { str : float, }
        robot_obs = robot.get_observation()
        # print("robot_obs", robot_obs)

        images, robot_state = process_observation(
            robot_obs, 
            env_cfg.joint_names, 
            env_cfg.camera_keys, 
            normalizer, 
            device
        ) # images: (B, num_cameras, 3, H, W)  robot_state: (B, state_dim)
        
        
        raw_actions = model.select_action(images, robot_state) # (B, action_dim)
        action = normalizer.denormalize_action(raw_actions)          
            
        # print("action", action.shape)
        robot_action = custom_make_robot_action(action, env_cfg.joint_names)

        robot.send_action(robot_action)

    print("Episode finished!")


    robot.disconnect()

def main(env_cfg : SO101ImitationLearningPlayEnvCfg, policy_cfg : ActCfg):

    env_cfg.seed = policy_cfg.seed
    torch.manual_seed(env_cfg.seed)

    # specify directory of experiments
    log_root_path = os.path.join("logs", "il", "so101")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    os.makedirs(log_root_path, exist_ok=True)


    if policy_cfg.load_dir is not None:
        log_dir = policy_cfg.load_dir
    else:
        existing_dirs = [
            d for d in os.listdir(log_root_path)
            if os.path.isdir(os.path.join(log_root_path, d))
        ]

        if len(existing_dirs) == 0:
            raise FileNotFoundError(
                f"No existing policy directories found in {log_root_path}. "
                "Cannot load a non-existent policy."
            )
        else:
            # pick the most recently modified directory
            latest_dir = max(
                existing_dirs,
                key=lambda d: os.path.getmtime(os.path.join(log_root_path, d))
            )
            log_dir = latest_dir
    
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    device = torch.device(env_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    play(env_cfg, policy_cfg, device)

    


if __name__ == "__main__":

    env_cfg = SO101ImitationLearningPlayEnvCfg()
    policy_cfg = ActCfg()
    main(env_cfg, policy_cfg)
