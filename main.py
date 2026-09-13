import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from models.iTransformer import Model as iTransformerModel
from preprocessing import fetch_stock_data, prepare_sequences

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def set_seed(seed):
    """Cố định Seed để kết quả có thể tái tạo (Reproducibility)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MockConfig:
    def __init__(self, seq_len, label_len, pred_len, num_variates):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = num_variates
        self.dec_in = num_variates
        self.c_out = num_variates
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.dropout = 0.1
        self.embed = "timeF"
        self.freq = "d"
        self.activation = "gelu"
        self.output_attention = False
        self.use_norm = True
        self.class_strategy = "projection"

# ---------------------------------------------------------
# Training, Evaluation & Baseline
# ---------------------------------------------------------
def naive_baseline(X_enc, Y_true, scaler, pred_len, close_idx=3):
    """
    Naive Baseline: Prediction[t] = Close[t-1]
    Dùng giá trị Close ở bước thời gian cuối cùng của X_enc làm dự báo cho toàn bộ horizon.
    """
    B, _, N = X_enc.shape
    
    # Lấy giá trị cuối cùng của lịch sử (t-1)
    last_known_scaled = X_enc[:, -1, :].numpy() # shape: [B, N]
    last_known_unscaled = scaler.inverse_transform(last_known_scaled)
    last_close = last_known_unscaled[:, close_idx] # shape: [B]
    
    # Lặp lại giá trị này cho toàn bộ số ngày cần dự báo (pred_len)
    preds_close = np.repeat(last_close[:, np.newaxis], pred_len, axis=1) # shape: [B, pred_len]
    
    # Lấy Ground Truth
    y_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N).numpy()).reshape(B, pred_len, N)
    actual_close = y_unscaled[:, :, close_idx]
    
    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))
    
    return preds_close, actual_close, mae, rmse

def train_model_with_early_stopping(model, train_loader, val_loader, epochs=50, lr=0.001, device="cpu", patience=5, close_idx=3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for batch_x_enc, batch_x_dec, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x_enc, batch_x_dec, batch_y = batch_x_enc.to(device), batch_x_dec.to(device), batch_y.to(device)
            batch_x_mark_enc = torch.zeros(batch_x_enc.shape[0], batch_x_enc.shape[1], 4).to(device)
            batch_x_mark_dec = torch.zeros(batch_x_dec.shape[0], batch_x_dec.shape[1], 4).to(device)

            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            outputs = outputs[:, -batch_y.shape[1]:, :]

            # CHỈ TỐI ƯU TRÊN CỘT CLOSE
            loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_enc, batch_x_dec, batch_y in val_loader:
                batch_x_enc, batch_x_dec, batch_y = batch_x_enc.to(device), batch_x_dec.to(device), batch_y.to(device)
                batch_x_mark_enc = torch.zeros(batch_x_enc.shape[0], batch_x_enc.shape[1], 4).to(device)
                batch_x_mark_dec = torch.zeros(batch_x_dec.shape[0], batch_x_dec.shape[1], 4).to(device)

                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                outputs = outputs[:, -batch_y.shape[1]:, :]
                
                v_loss = criterion(outputs[:, :, close_idx], batch_y[:, :, close_idx])
                val_loss += v_loss.item()
                
        val_loss /= len(val_loader)
        
        # --- EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # Trả model về trọng số tốt nhất trước khi thoát
                model.load_state_dict(best_model_wts)
                break

def evaluate_and_predict(model, X_enc, X_dec, Y_true, scaler, pred_len, close_idx=3, device="cpu"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_enc, X_dec = X_enc.to(device), X_dec.to(device)
        X_mark_enc = torch.zeros(X_enc.shape[0], X_enc.shape[1], 4).to(device)
        X_mark_dec = torch.zeros(X_dec.shape[0], X_dec.shape[1], 4).to(device)

        preds = model(X_enc, X_mark_enc, X_dec, X_mark_dec)
        preds = preds[:, -pred_len:, :].cpu()

    B, _, N = preds.shape
    preds_unscaled = scaler.inverse_transform(preds.reshape(-1, N)).reshape(B, pred_len, N)
    y_test_unscaled = scaler.inverse_transform(Y_true.reshape(-1, N).numpy()).reshape(B, pred_len, N)

    preds_close = preds_unscaled[:, :, close_idx]
    actual_close = y_test_unscaled[:, :, close_idx]

    mae = mean_absolute_error(actual_close.flatten(), preds_close.flatten())
    rmse = np.sqrt(mean_squared_error(actual_close.flatten(), preds_close.flatten()))

    return preds_close, actual_close, mae, rmse

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device: {device} ---")

    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2026-01-01"
    SEQ_LEN = 60
    LABEL_LEN = 30
    BATCH_SIZE = 32
    EPOCHS = 100 # Cứ để 100, Early Stopping sẽ tự cắt
    SEEDS = [42, 2024, 8888]
    PRED_LENS = [1, 5]

    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    NUM_VARIATES = df.shape[1]
    CLOSE_IDX = 3 

    for PRED_LEN in PRED_LENS:
        print(f"\n=======================================================")
        print(f" EXPERIMENT: PREDICTION LENGTH = {PRED_LEN} ")
        print(f"=======================================================")

        X_enc, X_dec, Y, scaler, train_end, val_end = prepare_sequences(
            df, SEQ_LEN, LABEL_LEN, PRED_LEN, train_ratio=0.7, val_ratio=0.15
        )

        # Train / Val / Test Splits
        X_enc_train, X_dec_train, Y_train = X_enc[:train_end], X_dec[:train_end], Y[:train_end]
        X_enc_val, X_dec_val, Y_val = X_enc[train_end:val_end], X_dec[train_end:val_end], Y[train_end:val_end]
        X_enc_test, X_dec_test, Y_test = X_enc[val_end:], X_dec[val_end:], Y[val_end:]

        train_loader = DataLoader(TensorDataset(X_enc_train, X_dec_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_enc_val, X_dec_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

        # --- 1. RUN NAIVE BASELINE ---
        naive_preds, actuals, naive_mae, naive_rmse = naive_baseline(X_enc_test, Y_test, scaler, PRED_LEN, CLOSE_IDX)
        print(f"[NAIVE BASELINE] MAE: {naive_mae:.4f} | RMSE: {naive_rmse:.4f}")

        # --- 2. RUN iTRANSFORMER ACROSS SEEDS ---
        itrans_maes, itrans_rmses = [], []
        best_preds = None # Dùng để plot

        for seed in SEEDS:
            set_seed(seed)
            configs = MockConfig(seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN, num_variates=NUM_VARIATES)
            model = iTransformerModel(configs)
            
            train_model_with_early_stopping(
                model, train_loader, val_loader, epochs=EPOCHS, lr=0.001, 
                device=device, patience=5, close_idx=CLOSE_IDX
            )

            i_preds, _, i_mae, i_rmse = evaluate_and_predict(
                model, X_enc_test, X_dec_test, Y_test, scaler, PRED_LEN, CLOSE_IDX, device=device
            )
            
            itrans_maes.append(i_mae)
            itrans_rmses.append(i_rmse)
            best_preds = i_preds # Giữ lại dự đoán của seed cuối cùng để vẽ chart
            
            print(f" - Seed {seed:4d} -> MAE: {i_mae:.4f} | RMSE: {i_rmse:.4f}")

        # Tính Mean và Std
        mean_mae, std_mae = np.mean(itrans_maes), np.std(itrans_maes)
        mean_rmse, std_rmse = np.mean(itrans_rmses), np.std(itrans_rmses)
        
        print(f"\n[iTRANSFORMER FINAL] MAE: {mean_mae:.4f} ± {std_mae:.4f} | RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

        # --- 3. PLOTTING ---
        plt.figure(figsize=(12, 6))
        
        if PRED_LEN == 1:
            # Nếu dự báo 1 ngày, nối các điểm lại và vẽ 100 ngày cuối
            plot_range = 100
            plt.plot(actuals.flatten()[-plot_range:], label="Actual Close", color="black", marker=".")
            plt.plot(naive_preds.flatten()[-plot_range:], label="Naive Baseline", color="green", linestyle="--", alpha=0.7)
            plt.plot(best_preds.flatten()[-plot_range:], label=f"iTransformer", color="red", alpha=0.8)
            plt.title(f"{TICKER} - 1-Day Ahead Prediction (Last {plot_range} days)")
        else:
            # Nếu dự báo 5 ngày, lấy sample cuối cùng để vẽ nguyên một đoạn 5 ngày
            sample_idx = -1
            # Vẽ lịch sử trước đó (từ X_enc_test)
            hist_unscaled = scaler.inverse_transform(X_enc_test[sample_idx].numpy())[:, CLOSE_IDX]
            time_hist = np.arange(SEQ_LEN)
            time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)
            
            plt.plot(time_hist, hist_unscaled, label="Historical Close", color="gray")
            plt.plot(time_pred, actuals[sample_idx], label="Actual Future", color="black", marker="o")
            plt.plot(time_pred, naive_preds[sample_idx], label="Naive Baseline", color="green", linestyle="--", marker="x")
            plt.plot(time_pred, best_preds[sample_idx], label="iTransformer", color="red", marker="s")
            plt.title(f"{TICKER} - {PRED_LEN}-Days Horizon Prediction")

        plt.xlabel("Time Steps")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"prediction_{PRED_LEN}day_chart.png", dpi=300)
        plt.close() # Đóng plot để không bị ghi đè sang vòng lặp sau
        print(f"[+] Đã lưu biểu đồ: prediction_{PRED_LEN}day_chart.png")