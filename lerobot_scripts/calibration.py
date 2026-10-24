from lerobot.robots.so_follower import  SO101Follower
from lerobot.teleoperators.so_leader import SO101Leader


from env_cfg import SO101TeleopEnvCfg

env_cfg = SO101TeleopEnvCfg()


print("Calibration Follower: ", env_cfg.robot_id) 

follower = SO101Follower(env_cfg.robot_cfg)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()

print("Calibration Leader: ", env_cfg.teleop_id) 

leader = SO101Leader(env_cfg.teleop_cfg)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()