import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

from act_transformer import DETRDecoderLayer, DETREncoderLayer


class CVAEEncoder(nn.Module):
    """
    Conditional Variational Autoencoder Encoder
    """
    def __init__(self, state_dim, action_dim, chunk_size, dim_feedforward=3200,d_model=256,
                 nhead=8, num_layers=4, latent_dim=32, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=chunk_size + 2, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, latent_dim * 2)
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, robot_state: torch.Tensor, actions: torch.Tensor, actions_padding_mask: torch.Tensor):
        """
        Args:
            robot_state : (B, state_dim)
            actions     : (B, chunk_size, action_dim)
            action_mask : (B, chunk_size) bool — False(0) for valid, True(1) for padding
        """
        B = robot_state.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        state_token = self.state_proj(robot_state).unsqueeze(1)

        action_token = self.action_proj(actions)
        action_token = self.pos_enc(action_token)
        
        tokens = torch.cat([cls_token, state_token, action_token], dim=1) # (B, chunk_size + 2, d_model)

        full_mask = torch.cat(
            [
                torch.zeros((B, 2), dtype=torch.bool, device=actions.device),
                actions_padding_mask
            ],
            dim=1
        ) # (B, chunk_size + 2)

        out = self.encoder(tokens,src_key_padding_mask=full_mask)
        params = self.to_latent(out[:, 0])
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var


class ACTEncoder(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int,
                 num_layers: int, 
                 dropout: float = 0.1,
                 batch_first: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            DETREncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        src: torch.Tensor,                          # (B, N, d_model)
        src_pos: torch.Tensor | None = None,        # (1, N, d_model)
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            src = layer(src, src_pos=src_pos, src_key_padding_mask=src_key_padding_mask)

        return src                                  # (B, N, d_model)
        

class ACTDecoder(nn.Module):
    def __init__(self, 
                 action_dim, 
                 chunk_size,
                 d_model=256, 
                 nhead=8,
                 dim_feedforward = 3200,  
                 num_layers=7, 
                 dropout=0.1, 
                 batch_first=True):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.query_pos_embed = nn.Embedding(chunk_size, d_model)
        
        # self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=chunk_size + 2, dropout=dropout)
        
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model, 
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout, 
        #     batch_first=batch_first,
        # )

        self.layers = nn.ModuleList([
            DETRDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model) 
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        enc_out: torch.Tensor,                           # (B, N_mem, d_model)
        enc_out_pos: torch.Tensor | None = None,         # (1, N_mem, d_model)
        enc_out_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        B = enc_out.size(0)

        tgt_pos = self.query_pos_embed.weight.unsqueeze(0)                              # (1, chunk_size, d_model)
        # tgt = torch.zeros((B, self.chunk_size, self.d_model)).devicesame(tgt_pos)       # (B, chunk_size, d_model)

        tgt = torch.zeros(
            (B, self.chunk_size, self.d_model),
            dtype=tgt_pos.dtype,
            device=tgt_pos.device,
        )

        for layer in self.layers:
            tgt = layer(
                tgt, 
                enc_out,
                tgt_pos=tgt_pos,
                enc_out_pos=enc_out_pos,
                enc_out_key_padding_mask=enc_out_key_padding_mask,
            )                                                                   # (B, chunk_size, d_model)
                              

        return self.action_head(self.norm(tgt))                                 # (B, chunk_size, action_dim)


class VisualBackbone(nn.Module):
    """
    Resnet
    conv1 
    layer1
    layer2
    layer3
    layer4
    avgpool   
    fc
    
    """
    def __init__(self, config):
        super().__init__()

        resnet_model = getattr(torchvision.models, config.vision_backbone_name)(
            replace_stride_with_dilation=[False, False, False],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )

        self.resnet = IntermediateLayerGetter(
            resnet_model, return_layers={"layer4": "feature_map"}
        )

        # ResNet18/34 → 512 channels, ResNet50/101 → 2048 channels
        num_channels = 512 if config.vision_backbone_name in ("resnet18", "resnet34") else 2048
        self.num_channels = num_channels

        assert config.d_model == num_channels, (
            f"d_model ({config.d_model}) must equal backbone output channels ({num_channels}). "
            f"For {config.vision_backbone_name} set d_model={num_channels}."
        )


    def forward(self, images):
        """
        ResNet is fully convolutional — works with any input resolution.
        Output spatial size = input_size / 32 (ResNet's total stride).
        
        Examples:
            Input (B, 3, 240, 320) → Output (B, 80, 512)    # 8x10 = 80 tokens
            Input (B, 3, 480, 640) → Output (B, 300, 512)   # 15x20 = 300 tokens
        
        Args:
            images: (B, 3, H, W) where H, W are arbitrary (preferably divisible by 32)
        
        Returns:
            # tokens: (B, H'x W', 512) where H'= H/32, W'= W/32
            (B, num_channels, H', W')
        """
        features = self.resnet(images)["feature_map"]   # (B, num_channels, H', W')
        # B, C, H, W = features.shape
        # out = features.flatten(2).permute(0, 2, 1)     # (B, H' * W', num_channels)
        # return out
        return features


class SinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, d_model: int, temperature: int = 10_000):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.temperature = temperature
        self._two_pi = 2 * math.pi


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, d_model, H, W)
        """
        _, _, H, W = x.shape
        device = x.device

        y_range = torch.arange(H, device=device, dtype=torch.float32)   # (H,)
        x_range = torch.arange(W, device=device, dtype=torch.float32)   # (W,)


        eps = 1e-6
        y_range = y_range / (H + eps)  * self._two_pi                   # (H,)
        x_range = x_range  / (W + eps) * self._two_pi                   # (W,)


        dim_t = torch.arange(self.d_model // 2, device=device, dtype=torch.float32)         # (d_model // 2,)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 2))                # (d_model // 2,)


        pos_x = x_range[:, None] / dim_t    # (H, d_model//2)
        pos_y = y_range[:, None] / dim_t    # (W, d_model//2)


        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos(),), dim=-1,).flatten(-2)  # (W, d_model // 2)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos(),), dim=-1,).flatten(-2)  # (H, d_model // 2)


        pos_x = pos_x[None, :, :].expand(H, W, self.d_model // 2,)      # (1, W, d_model // 2) -> (H, W, d_model // 2)
        pos_y = pos_y[:, None, :].expand(H, W, self.d_model // 2,)      # (H, 1, d_model // 2) -> (H, W, d_model // 2)
             


        pos = torch.cat([pos_y, pos_x], dim=-1)  # (H, W, d_model)
        pos = pos.permute(2, 0, 1).unsqueeze(0) # (1, d_model, H, W)

        return pos
    


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.
    Input: 
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)   # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ) # [d_model // 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape [B, T, d_model]."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Normalizer:
    """
    Handles normalization/denormalization using dataset statistics.
    """
    def __init__(self, stats: dict):
        self.stats = stats

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) → normalized"""
        mean = torch.tensor(self.stats["observation.state"]["mean"], 
                           dtype=state.dtype, device=state.device)
        std  = torch.tensor(self.stats["observation.state"]["std"],
                           dtype=state.dtype, device=state.device)
        return (state - mean) / (std + 1e-8)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """(B, chunk_size, action_dim) → normalized"""
        mean = torch.tensor(self.stats["action"]["mean"],
                           dtype=action.dtype, device=action.device)
        std  = torch.tensor(self.stats["action"]["std"],
                           dtype=action.dtype, device=action.device)
        return (action - mean) / (std + 1e-8)

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """(B, chunk_size, action_dim) normalized → original scale"""
        mean = torch.tensor(self.stats["action"]["mean"],
                           dtype=action_norm.dtype, device=action_norm.device)
        std  = torch.tensor(self.stats["action"]["std"],
                           dtype=action_norm.dtype, device=action_norm.device)
        return action_norm * std + mean

    def normalize_image(self, image: torch.Tensor, camera_key: str) -> torch.Tensor:
        """
        (B, C, H, W) or (C, H, W) → normalized

        [3, h=240, width=320]
  
        
        Args:
            image: raw image in [0, 1] range
            camera_key: e.g. "observation.images.front"
        """
        if camera_key not in self.stats:
            # Fallback to no normalization if stats not available
            return image
        
        # Stats are (3, 1, 1) — broadcast over (C, H, W)
        mean = torch.tensor(self.stats[camera_key]["mean"],
                           dtype=image.dtype, device=image.device).squeeze()  # (3,)
        std  = torch.tensor(self.stats[camera_key]["std"],
                           dtype=image.dtype, device=image.device).squeeze()   # (3,)
        
        # Reshape to (3, 1, 1) for broadcasting
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)
        
        return (image - mean) / (std + 1e-8)
    

class TemporalEnsembler:
    """
    Maintains a rolling buffer of overlapping action chunks and combines them
    via exponential weighting as described in the ACT paper:

        w_i = exp(-m * i),  where i=0 is the OLDEST prediction for this timestep.

    At each call to `update(action_chunk)`:
      1. The new chunk is pushed into the buffer.
      2. All buffered predictions that cover the *current* timestep are gathered.
      3. A weighted average is computed and the current-step action is returned.
      4. The buffer advances by one step.
    """

    def __init__(self, chunk_size: int, device: torch.device, temporal_ensemble_coeff: float = 0.1):
        """
        Args:
            chunk_size                  : number of steps per action chunk (T)
            action_dim                  : dimensionality of a single action (A)
            temporal_ensemble_coeff     : exponential decay rate; smaller → slower incorporation
                                              of new observations (paper default ≈ 0.1)
        """
        self.chunk_size = chunk_size
        self.device = device

        weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size, device=device))
        self.ensemble_weights = weights
        self.ensemble_weights_cumsum = torch.cumsum(weights, dim=0)

        self.reset()

    def reset(self):
        self.ensembled_actions = None        # (B, chunk_size, action_dim)
        self.ensembled_actions_count = None  # (chunk_size, 1)
        self.head = 0
        self.initialized = False

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions : (B, chunk_size, action_dim)
        Returns:
            action  : (B, action_dim)
        """
        B, k, action_dim = actions.shape

        if not self.initialized:
            self.ensembled_actions = actions.clone()           # (B, k, action_dim)
            self.ensembled_actions_count = torch.ones(
                k, 1, dtype=torch.long, device=self.device
            )
            self.initialized = True
        else:
            slots = (self.head + torch.arange(k - 1, device=self.device)) % self.chunk_size
            counts = self.ensembled_actions_count[slots] - 1  # (k-1, 1)

            w_old = self.ensemble_weights_cumsum[counts]      # (k-1, 1)
            w_new = self.ensemble_weights[counts + 1]         # (k-1, 1)
            w_tot = self.ensemble_weights_cumsum[counts + 1]  # (k-1, 1)

            # (B, k-1, action_dim) — broadcast over batch dim automatically
            self.ensembled_actions[:, slots] = (
                self.ensembled_actions[:, slots] * w_old
                + actions[:, :-1] * w_new
            ) / w_tot

            self.ensembled_actions_count[slots] = torch.clamp(
                self.ensembled_actions_count[slots] + 1, max=self.chunk_size
            )

            new_slot = (self.head + k - 1) % self.chunk_size
            self.ensembled_actions[:, new_slot] = actions[:, -1]   # (B, action_dim)
            self.ensembled_actions_count[new_slot] = 1

        action = self.ensembled_actions[:, self.head]              # (B, action_dim)
        self.head = (self.head + 1) % self.chunk_size

        return action
