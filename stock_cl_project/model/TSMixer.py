import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout),
        )

    def forward(self, x):
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.num_variates = configs.enc_in
        self.representation_dim = configs.seq_len * configs.enc_in
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forward_features(self, x_enc):
        for block in self.model:
            x_enc = block(x_enc)
        return x_enc  # [B, seq_len, num_variates]

    def encode(self, x_enc, x_mark_enc=None, mask=None):
        features = self.forward_features(x_enc)
        return features.flatten(start_dim=1)

    def forecast(self, x_enc, x_mark_enc=None, mask=None):
        features = self.forward_features(x_enc)
        return self.projection(features.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, mask)
            return dec_out[:, -self.pred_len :, :]
        raise ValueError("Only forecast tasks implemented yet")
