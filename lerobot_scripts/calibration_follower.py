from lerobot.robots.so_follower import  SO101Follower


from env_cfg import SO101EnvCfg

env_cfg = SO101EnvCfg()

follower = SO101Follower(env_cfg.robot_cfg)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()