"""
"""

import os
import psutil
from pathlib import Path
from tqdm import tqdm

import time


from datetime import datetime


import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader





from utils import dump_yaml, Logger
from act_policy import ACT
from policy_cfg import  ActCfg
from env_cfg import SO101ImitationLearningEnvCfg

from dataset_manager import DatasetManager


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


# ─────────────────────────────────────────────────────────────────────
# Train / val steps
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model:    ACT,
    loader:   DataLoader,
    data_mngr: DatasetManager,
    device:   torch.device,
) -> dict:
    
    model.eval()
    totals = {"loss": 0., "l1": 0., "kl": 0.}

    for batch in loader:

        images, robot_state, actions, action_mask = data_mngr.process_batch(
            batch, device
        )
        out = model(images, robot_state, actions, action_mask=action_mask)

        totals["loss"]  += out["loss"].item()
        totals["l1"] += out["loss_l1"].item()
        totals["kl"]    += out["loss_kl"].item()


    n = len(loader)
    val_metrics = {k: v / n for k, v in totals.items()}


    return val_metrics



def training_loop(
    model:     ACT,
    train_loader:   DataLoader,
    data_mngr: DatasetManager,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    logger:    Logger,
    device:    torch.device,
    log_interval:  int = 100,
    save_interval: int = 2000,
    max_steps:     int = 10000,
    checkpoint_dir: Path = "",
    resume_step: int = 0,
    
) -> dict:
       
    best_val_loss = float("inf")
    epoch_size = len(train_loader)
    train_iter = cycle(train_loader)

    print("epoch_size:", epoch_size)
    

    # model.train()

    for step in range(resume_step, max_steps):
        model.train()
        start_time = time.perf_counter()

        batch = next(train_iter)

        images, robot_state, actions, action_mask = data_mngr.process_batch(
            batch, device
        )

        out = model(images, robot_state, actions, action_mask=action_mask)

        
        out["loss"].backward()
        # nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        

        duration = time.perf_counter() - start_time

        # ---- metrics ----
        metrics = {
            "loss": out["loss"].item(),
            "l1": out["loss_l1"].item(),
            "kl": out["loss_kl"].item(),
            "time": duration,
        }


        # Log per-step: global_step + step
        if step % log_interval == 0:
            epoch = step /epoch_size
            print(
                f"----------------- \n"
                f"step={step} \n"
                f"epoch={epoch:.2f} \n"
                f"loss={metrics['loss']:.4f}  \n"
                f"l1={metrics['l1']:.4f}  \n"
                f"kl={metrics['kl']:.4f} \n"
                f"time={metrics['time']:.4f} \n"
            )
            ram = psutil.virtual_memory()
            print(f"RAM used: {ram.used / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB")

            logger.log(
                {
                    "train/loss": metrics["loss"],
                    "train/l1":  metrics["l1"],
                    "train/kl": metrics["kl"],
                },
                step=step,
                prefix="step",
            )


        # ── Periodic checkpoint ───────────────────────────────────────
        if step > 0 and step % save_interval == 0:
            path = Path(checkpoint_dir) / f"act_step_{step:06d}.pth"
            
            save_model(model, optimizer, scheduler, path)


        # # ── Validation Dataset ────────────────────────────────────────────
        # if step % epoch_size == 0 and step > 0: 

        #     val_metrics = validate(model, val_loader, data_mngr, device)

        #     # ── Save best ─────────────────────────────────────────────────
        #     if val_metrics["loss"] < best_val_loss:
        #         best_val_loss = val_metrics["loss"]
        #         path = Path(checkpoint_dir) / "act_best.pth"
        #         path.parent.mkdir(parents=True, exist_ok=True)

        #         save_model(model, optimizer, scheduler, path)


        scheduler.step()


    # Save final checkpoint
    path = Path(checkpoint_dir) / f"act_step_{step:06d}.pth"
    save_model(model, optimizer, scheduler, path)
    print(f"Saved final checkpoint at step {step}")

    return best_val_loss


def save_model(model, optimizer, scheduler, path):

    save_dict = {
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), 
    }
    
    torch.save(save_dict, path)





# ─────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────

def train(env_cfg : SO101ImitationLearningEnvCfg, policy_cfg : ActCfg, device : torch.device):

    data_manager = DatasetManager(
        repo_id=env_cfg.dataset_name, 
        local_path=env_cfg.dataset_local_path,
        episodes=env_cfg.dataset_episodes
        )
    # train_loader, val_loader = data_manager.build_dataloaders(policy_cfg, device)

    train_loader = data_manager.build_dataloaders(policy_cfg, device)

      

    # ── Model ─────────────────────────────────────────────────────────
    model = ACT(policy_cfg).to(device)

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())

    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:50} {p.numel():,}")

    print(f"Trainable parameters : {n_trainable_params:,}")
    print(f"Total parameters     : {n_total_params:,}")

    # ── Optimizer — lower lr for pretrained backbone ───────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    

    optimizer = AdamW(
        trainable_params,
        lr=policy_cfg.lr,  
        weight_decay=policy_cfg.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=policy_cfg.max_steps, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────
    

    logger = Logger(log_dir=policy_cfg.log_dir, use_wandb=policy_cfg.use_wandb)


    try:
        training_loop(
                model=model, 
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                data_mngr=data_manager, 
                logger=logger,
                device=device,
                log_interval=policy_cfg.log_interval, 
                save_interval=policy_cfg.save_interval,
                max_steps=policy_cfg.max_steps,
                checkpoint_dir=Path(env_cfg.log_dir) / policy_cfg.save_dir,
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
    


    logger.close()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main(env_cfg : SO101ImitationLearningEnvCfg, policy_cfg : ActCfg):
    """Train an imitation policy."""

    if policy_cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    env_cfg.seed = policy_cfg.seed

    torch.manual_seed(env_cfg.seed)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "il", "so101")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if policy_cfg.run_name:
        log_dir += f"_{policy_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env_cfg.log_dir = log_dir

    # if policy_cfg.resume:
    #     resume_path = get_checkpoint_path(
    #         log_dir, 
    #         policy_cfg.load_run, 
    #         policy_cfg.load_checkpoin
    #     )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "policy.yaml"), policy_cfg)

    # ── Device
    device = torch.device(env_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    # create checkpoint directory
    checkpoint_dir = os.path.join(log_dir, policy_cfg.save_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
  

    train(env_cfg, policy_cfg, device)



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    env_cfg = SO101ImitationLearningEnvCfg()
    policy_cfg = ActCfg()
    main(env_cfg, policy_cfg)
    