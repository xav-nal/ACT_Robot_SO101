from dataclasses import dataclass


@dataclass
class PolicyCfg():
    device:                 str = "cuda"
    seed:                   int = 42
  
    run_name:               str = ""
    
    load_dir:               str = ""
    checkpoint_to_load:     str = "act_best.pth"

    cudnn_deterministic:    bool = False
    chunk_size:             int=100 # 50

    # Training
    lr:                     float = 1e-5 
    weight_decay:           float = 1e-4
    grad_clip:              int   = 40

    # num_epochs:             int   = 20
    val_ratio:              float =  0.1
    val_n_episodes:         int = 2
    
    log_interval:           int = 100

    save_interval:          int = 50000
    save_dir:               str = "checkpoints"

    max_steps:              int = 100000

    batch_size:             int = 32 # 8 # 128        #  128
    num_workers:            int = 4


    log_dir: str = ""
    use_wandb: bool = True



@dataclass
class ActCfg(PolicyCfg):

    state_dim:          int = 6
    action_dim:         int = 6

    # Policy
    dim_feedforward:    int = 3200
    d_model:            int = 512
    dropout:            float = 0.1

    kl_weight:          float  = 10.0
    latent_dim:         int = 32
    
    
    n_action_steps:         int = 100 # no use for the moment
    num_decoder_layers:     int = 1 # 7
    num_encoder_layers:     int = 4
    nhead:                  int = 8
    n_obs_steps:            int = 1 # no use for the moment
    n_vae_encoder_layers:   int = 4


    pretrained_backbone_weights:    str = "IMAGENET1K_V1"
    vision_backbone_name:           str = "resnet18"  # "resnet34"
    
    
    temporal_ensemble_coeff: float = 0.1


    
    # Misc
    # mixed_precision:     str   = "fp16"  # "no", "fp16", "bf16"