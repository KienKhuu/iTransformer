import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_variates, seq_len, pred_len, hidden_size=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_variates = num_variates

        self.lstm = nn.LSTM(
            input_size=num_variates,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size, pred_len * num_variates)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        lstm_out, _ = self.lstm(x)  # [B, L, H]
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)  # [B, pred_len * N]
        return out.reshape(-1, self.pred_len, self.num_variates)
