import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, cfg, obs_space, timing):
        super().__init__()
        self.obs_space = obs_space
        self.cfg = cfg

        self.transformer = nn.Transformer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_attention_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )

        self.fc = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, obs_dict):
        x = obs_dict['obs']
        x = self.transformer(x)
        x = self.fc(x)
        return x

def make_transformer_encoder(cfg, obs_space, timing):
    return TransformerEncoder(cfg, obs_space, timing)