# transformer_model.py

import torch
import torch.nn as nn

class ConfidenceTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.conf_proj = nn.Linear(1, embed_dim)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x, conf_score=None):
        x = self.embedding(x)  # (B, L, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        x = x.mean(dim=1)  # (B, D)
        if conf_score is not None:
            conf_embed = self.conf_proj(conf_score)  # (B, D)
            x = x + conf_embed
        return self.fc_out(x)
