import os
import yaml
from dataclasses import asdict, is_dataclass
from pathlib import Path


import torch
from torch.utils.tensorboard import SummaryWriter


import wandb


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"

    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    # convert data into serializable dict
    def convert(obj):
        if is_dataclass(obj):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    data_dict = convert(data)

    # write yaml
    with open(filename, "w") as f:
        yaml.dump(
            data_dict,
            f,
            default_flow_style=False,
            sort_keys=sort_keys,
            allow_unicode=True
        )



class Logger:
    def __init__(self, 
                 log_dir, 
                 use_wandb=False, 
                 project_name="act-so101", 
                 run_name="train"):
        
        self.writer = SummaryWriter(log_dir)

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                dir=log_dir
            )




    def log(self, metrics: dict, step: int, prefix: str = ""):
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k

            # TensorBoard
            self.writer.add_scalar(tag, v, step)

            # WandB
            if self.use_wandb:
                wandb.log({tag: v}, step=step)


    def close(self):
        self.writer.close()
        if self.use_wandb:
            wandb.finish()