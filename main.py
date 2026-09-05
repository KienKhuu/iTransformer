import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

# Import the official models from the cloned repository
from model.iTransformer import Model as iTransformerModel
from preprocessing import fetch_stock_data, prepare_sequences


# ---------------------------------------------------------
# Mock Configuration Class
# ---------------------------------------------------------
class MockConfig:
    """
    Mocks the argparse configuration expected by the official models.
    Contains all attributes required for initialization.
    """

    def __init__(self, seq_len, label_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = num_variates  # Input features
        self.dec_in = num_variates  # Decoder input features
        self.c_out = num_variates  # Output features
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2  # Encoder layers
        self.d_layers = 1  # Decoder layers (used by Transformer)
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.dropout = 0.1
        self.embed = "timeF"  # Embedding type
        self.freq = "d"  # Frequency
        self.activation = "gelu"
        self.output_attention = False
        # Specific to iTransformer
        self.use_norm = True
        self.class_strategy = "projection"


# ---------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------
def train_model(model, train_loader, epochs=15, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x_enc, batch_x_dec, batch_y in train_loader:
            optimizer.zero_grad()

            # Create dummy time markers (since we skip date parsing for simplicity)
            batch_x_mark_enc = torch.zeros(
                batch_x_enc.shape[0], batch_x_enc.shape[1], 4
            )
            batch_x_mark_dec = torch.zeros(
                batch_x_dec.shape[0], batch_x_dec.shape[1], 4
            )

            # Official models expect: x_enc, x_mark_enc, x_dec, x_mark_dec
            outputs = model(
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec
            )

            # Extract the prediction part (last pred_len steps)
            if self_attention_model_check(model):
                outputs = outputs[:, -batch_y.shape[1] :, :]
            else:
                outputs = outputs[:, -batch_y.shape[1] :, :]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def evaluate_and_predict(model, X_enc, X_dec, Y_true, scaler, pred_len, close_idx=3):
    model.eval()
    with torch.no_grad():
        # Dummy time markers
        X_mark_enc = torch.zeros(X_enc.shape[0], X_enc.shape[1], 4)
        X_mark_dec = torch.zeros(X_dec.shape[0], X_dec.shape[1], 4)

        preds = model(X_enc, X_mark_enc, X_dec, X_mark_dec)
        preds = preds[:, -pred_len:, :]  # Get only the future prediction window

    B, _, N = preds.shape

    # Inverse transform
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, N)).reshape(
        B, pred_len, N
    )
    y_test_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N)).reshape(
        B, pred_len, N
    )

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_test_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))

    return preds_close, actual_close, mae, rmse


def self_attention_model_check(model):
    return True  # Helper just for logical layout


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Setup Parameters
    TICKER = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = "2023-01-01"
    SEQ_LEN = 96  # Standard look-back in iTransformer paper
    LABEL_LEN = 48  # Decoder overlap
    PRED_LEN = 24  # Forecast horizon
    BATCH_SIZE = 32

    # 2. Fetch Data
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = 3  # Position of 'Close' in OHLCV

    X_enc, X_dec, Y, scaler = prepare_sequences(df, SEQ_LEN, LABEL_LEN, PRED_LEN)

    # Train/Test Split
    split_idx = int(len(X_enc) * 0.8)
    X_enc_train, X_dec_train, Y_train = (
        X_enc[:split_idx],
        X_dec[:split_idx],
        Y[:split_idx],
    )
    X_enc_test, X_dec_test, Y_test = X_enc[split_idx:], X_dec[split_idx:], Y[split_idx:]

    train_dataset = TensorDataset(X_enc_train, X_dec_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Create Config and Initialize Models
    configs = MockConfig(
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        num_variates=NUM_VARIATES,
    )

    print("\n--- Training Official iTransformer ---")
    itransformer_model = iTransformerModel(configs)
    train_model(itransformer_model, train_loader, epochs=10)

    # 4. Evaluation
    print("\n--- Evaluation on Test Set ('Close' Price) ---")

    i_preds, actuals, i_mae, i_rmse = evaluate_and_predict(
        itransformer_model, X_enc_test, X_dec_test, Y_test, scaler, PRED_LEN, CLOSE_IDX
    )

    print(f"Official iTransformer -> MAE: {i_mae:.4f}, RMSE: {i_rmse:.4f}")

    # 5. Plotting
    sample_idx = 0

    plt.figure(figsize=(10, 6))

    # Thêm đường thực tế (Màu đen)
    plt.plot(actuals[sample_idx], label="Actual Close Price", marker="o", color="black")
    # Đường dự báo của iTransformer (Màu đỏ)
    plt.plot(i_preds[sample_idx], label="iTransformer", marker="s", color="red")

    plt.title(f"{TICKER} Future {PRED_LEN} Days Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
