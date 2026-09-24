import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_variates, seq_len, pred_len, hidden_size=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_variates = num_variates
        self.hidden_size = hidden_size
        self.representation_dim = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_variates,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_variates)

    def encode(self, x_enc, x_mark_enc=None, mask=None):
        lstm_out, _ = self.lstm(x_enc)
        return lstm_out[:, -1, :]  # [B, hidden_size]

    def forward(self, x_enc, x_mark_enc=None, mask=None):
        last_out = self.encode(x_enc, x_mark_enc, mask)
        out = self.fc(last_out)
        return out.reshape(-1, self.pred_len, self.num_variates)
