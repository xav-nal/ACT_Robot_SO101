
import torch
import torch.nn as nn
from torch.nn import functional as F


from act_utils import (
    SinusoidalPositionalEncoding,
    SinusoidalPositionEmbedding2d,
    CVAEEncoder,
    VisualBackbone,
    ACTEncoder,
    ACTDecoder,
    TemporalEnsembler,
)



from policy_cfg import ActCfg


class ACT(nn.Module):
    """
    me                          51,597,190
    lerobot numb parameters :   51,597,190 (52M)

    """
    def __init__(self, config: ActCfg):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight

        self.state_proj = nn.Linear(config.state_dim, config.d_model)
        self.latent_proj = nn.Linear(config.latent_dim, config.d_model)

        self.learn_pos_embed = nn.Embedding(2, config.d_model)


        # ── CVAE Encoder (training only) ──────────────────────────────
        self.cvae_encoder = CVAEEncoder(
            state_dim=config.state_dim, 
            action_dim=config.action_dim, 
            chunk_size=config.chunk_size,
            dim_feedforward=config.dim_feedforward, 
            d_model=config.d_model, 
            nhead=config.nhead,
            num_layers=config.num_encoder_layers, 
            latent_dim=config.latent_dim, 
            dropout=config.dropout,
        )


        # ── Visual backbone ───────────────────────────────────────────
        self.backbone = VisualBackbone(config)  
        self.visual_proj = nn.Conv2d(self.backbone.num_channels, config.d_model, kernel_size=1)
        self.visual_pos_embed = SinusoidalPositionEmbedding2d(d_model=config.d_model)


        # ── Transformer Encoder (policy encoder) ──────────────────────
        self.encoder = ACTEncoder(
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            num_layers=config.num_encoder_layers,
            dropout=config.dropout,
            batch_first=True
        )


        # ── Transformer Decoder (policy decoder) ──────────────────────
        self.decoder = ACTDecoder(
            action_dim=config.action_dim, 
            chunk_size=config.chunk_size,
            d_model=config.d_model, 
            nhead=config.nhead, 
            dim_feedforward=config.dim_feedforward, 
            num_layers=config.num_decoder_layers,
            dropout=config.dropout,
            batch_first=True
        )


        # ── Temporal ensembler (inference only) ──────────────────────────────
        self.temporal_ensembler = TemporalEnsembler(
            chunk_size=config.chunk_size,
            device=config.device,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,   
        )
 

    def reparameterize(self, mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)


    @staticmethod
    def kl_divergence(mu, log_var):
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    

    def masked_l1_loss(self, pred, target, mask):
        """
        prediction: (B, T, A)
        target:     (B, T, A)
        mask:       (B, T) ->  0(False) for valid, 1(True) for padding
        """
        loss = F.l1_loss(pred, target, reduction="none")  # (B, T, A)

        valid_mask =  (~mask).float().unsqueeze(-1)  # (B, T, 1)

        loss = loss * valid_mask

        return loss.sum() / valid_mask.sum().clamp(min=1.0)


    def build_token_inputs(self, images, robot_state, z):
        """
        Inputs:
            images      : (B, num_cameras, 3, H, W)
            robot_state : (B, state_dim)
            z           : (B, latent_dim)
        Outputs:
            tokens      : (B, N_mem, d_model)
                        N_mem = H'*W' * num_cameras + 1 (state) + 1 (z)
            pos_embed_tokens
        """
        tokens = []
        tokens_pos_embed = []

        num_camera = images.shape[1]

        for i in range(num_camera):
            visual_features = self.backbone(images[:,i])                                # (B, num_channels, H', W')
            visual_pos_embed = self.visual_pos_embed(visual_features)                   # (B, d_model, H', W')
            visual_tokens = self.visual_proj(visual_features)                           # (B, d_model, H', W')

            visual_tokens_flatten = visual_tokens.flatten(2).permute(0, 2, 1)           # (B, H' * W', d_model)
            visual_pos_embed_flatten = visual_pos_embed.flatten(2).permute(0, 2, 1)     # (B, H' * W', d_model)

            tokens.append(visual_tokens_flatten)
            tokens_pos_embed.append(visual_pos_embed_flatten)                           # (B, d_model, H', W')


        
        tokens.append(self.state_proj(robot_state).unsqueeze(1))           # (B, 1, d_model)
        tokens.append(self.latent_proj(z).unsqueeze(1))                    # (B, 1, d_model)


        #tokens_pos_embed.append(self.learn_pos_embed)
        tokens_pos_embed.append(self.learn_pos_embed.weight.unsqueeze(0))
        

        input_tokens = torch.cat(tokens, dim=1)                                 # (B, N_mem, d_model)
        pos_embed_tokens = torch.cat(tokens_pos_embed, dim=1)                   # (1, N_mem, d_model)

        # tokens = input_tokens + pos_embed_tokens                              # (B, N_mem, d_model)

        return input_tokens, pos_embed_tokens


    def forward(self, images, robot_state, actions, action_mask):
        """
        Args:
            images      : (B, num_cameras, 3, H, W)
            robot_state : (B, state_dim)
            actions     : (B, chunk_size, action_dim)
            action_mask : (B, chunk_size) bool — True(1) for padding, False(0) for valid
        """
        assert actions is not None
        assert action_mask is not None, "action_mask required for training"

        mu, log_var = self.cvae_encoder(robot_state, actions, action_mask)  # mu and log_var shape (B, latent_dim)
        z = self.reparameterize(mu, log_var) # (B, latent_dim)

        # memory = self.build_token_inputs(images, robot_state, z) 
        # tokens = self.build_token_inputs(images, robot_state, z)  # (B, 162,512 )
        input_tokens, pos_embed_tokens = self.build_token_inputs(images, robot_state, z)

        encoder_output = self.encoder(
            src=input_tokens,
            src_pos=pos_embed_tokens
        )               # (B, N_mem, d_model)

        pred_actions = self.decoder(
            enc_out=encoder_output,
            enc_out_pos=pos_embed_tokens,
        )      # (B, chunk_size, action_dim)

        loss_l1 = self.masked_l1_loss(pred_actions, actions, action_mask)
        loss_kl = self.kl_divergence(mu, log_var)
        loss = loss_l1 + self.kl_weight * loss_kl

        return dict(
            pred_actions=pred_actions,
            mu=mu, 
            log_var=log_var,
            loss=loss, 
            loss_l1=loss_l1, 
            loss_kl=loss_kl
        )


    @torch.no_grad()
    def predict(self, images, robot_state, sample_latent=False):
        """
        Inference-time prediction.
        
        Args:
            images:       (B, num_cameras, 3, H, W)
            robot_state:  (B, state_dim)
            sample_latent: if True, sample z ~ N(0,I); else z= zeros(B, lentt_dim)
        
        Returns:
            pred_actions: (B, chunk_size, action_dim)
        """
        B, device = robot_state.size(0), robot_state.device
        z = torch.randn(B, self.latent_dim, device=device) if sample_latent \
            else torch.zeros(B, self.latent_dim, device=device)
        
        input = self.build_token_inputs(images, robot_state, z)
        encoder_output = self.encoder(input)

        return self.decoder(encoder_output)
    

    @torch.no_grad()
    def select_action(self, images, robot_state, sample_latent=False):

        B, device = robot_state.size(0), robot_state.device

        z = (
            torch.randn(B, self.latent_dim, device=device)
            if sample_latent
            else torch.zeros(B, self.latent_dim, device=device)
        )

        # (B, chunk_size, action_dim)
        # action_chunk = self.decoder(self.build_token_inputs(images, robot_state, z))

        tokens = self.build_token_inputs(images, robot_state, z)
        encoder_output = self.encoder(tokens)
        action_chunk = self.decoder(encoder_output)

        # (B, action_dim) — weighted average over overlapping predictions
        #return action_chunk[:,0]
        return self.temporal_ensembler.update(action_chunk)