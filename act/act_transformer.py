
import torch
from torch import nn
from torch.nn import functional as F



class DETREncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int, 
                 dropout: float = 0.1,
                 batch_first: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,                              # (B, N, d_model)
        src_pos: torch.Tensor | None = None,            # (B, N, d_model)
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q = k = src if src_pos is None else src + src_pos

        src2, _ = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(src2))

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))

        return src


class DETRDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int, 
                 dropout: float = 0.1, 
                 batch_first: bool = True
        ):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                                  # (B, chunk_size, d_model)
        enc_out: torch.Tensor,                              # (B, N, d_model)
        tgt_pos: torch.Tensor | None = None,                # (1, chunk_size, d_model)
        enc_out_pos: torch.Tensor | None = None,            # (1, N, d_model)
        tgt_key_padding_mask: torch.Tensor | None = None,
        enc_out_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ── Self-attention ───────────────────────────────────────────
        q = k = tgt if tgt_pos is None else tgt + tgt_pos
        tgt2, _ = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # ── Cross-attention ──────────────────────────────────────────
        q    = tgt    if tgt_pos    is None else tgt    + tgt_pos
        k    = enc_out if enc_out_pos is None else enc_out + enc_out_pos

        tgt2, _ = self.cross_attn(q, k, value=enc_out, key_padding_mask=enc_out_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # ── FFN ──────────────────────────────────────────────────────
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt