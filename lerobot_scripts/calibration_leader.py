from lerobot.teleoperators.so_leader import SO101Leader


from env_cfg import SO101TeleopEnvCfg

env_cfg = SO101TeleopEnvCfg()

leader = SO101Leader(env_cfg.teleop_cfg)
leader.connect(calibrate=False)
leader.calibrate()
leader.disconnect()