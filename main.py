import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_preprocessing import fetch_stock_data, load_and_preprocess_data

# Note: Adjust the import based on how the original iTransformer is structured
from models.iTransformer import Model as iTransformer

# --- Configuration ---
TICKER = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2024-01-01'
CSV_PATH = 'data/stock_data.csv'
SEQ_LEN = 96        # Lookback window (e.g., past 96 days)
PRED_LEN = 24       # Prediction window (e.g., next 24 days)
FEATURES = 5        # O, H, L, C, V
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dummy Config Object (iTransformer requires a config object) ---
class Config:
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.enc_in = FEATURES
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 256
        self.dropout = 0.1
        self.activation = 'gelu'
        self.output_attention = False

def train_model():
    print(f"Using device: {DEVICE}")
    
    train_loader, test_loader, scaler = load_and_preprocess_data(
        CSV_PATH, SEQ_LEN, PRED_LEN, BATCH_SIZE
    )
    
    config = Config()
    model = iTransformer(config).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x, None, batch_y, None)
            
            # iTransformer predicts all features. We calculate loss on all of them.
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
        print(f"Epoch: {epoch+1}/{EPOCHS} | Train Loss: {np.mean(train_loss):.4f}")
        
    return model, test_loader, scaler

def evaluate_and_plot(model, test_loader, scaler):
    model.eval()
    predictions = []
    trues = []
    
    print("Evaluating Model...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            
            outputs = model(batch_x, None, None, None)
            
            predictions.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())
            
    predictions = np.concatenate(predictions, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # The 'Close' price is typically index 3 in ['Open', 'High', 'Low', 'Close', 'Volume']
    CLOSE_IDX = 3
    
    # Reshape and Inverse Transform to get actual stock prices
    pred_close = predictions[:, :, CLOSE_IDX].reshape(-1)
    true_close = trues[:, :, CLOSE_IDX].reshape(-1)
    
    # Create dummy arrays to use inverse_transform properly
    dummy_pred = np.zeros((len(pred_close), FEATURES))
    dummy_true = np.zeros((len(true_close), FEATURES))
    dummy_pred[:, CLOSE_IDX] = pred_close
    dummy_true[:, CLOSE_IDX] = true_close
    
    pred_real_prices = scaler.inverse_transform(dummy_pred)[:, CLOSE_IDX]
    true_real_prices = scaler.inverse_transform(dummy_true)[:, CLOSE_IDX]

    # --- Metrics Calculation ---
    mae = mean_absolute_error(true_real_prices, pred_real_prices)
    rmse = np.sqrt(mean_squared_error(true_real_prices, pred_real_prices))
    
    print("-" * 30)
    print(f"Evaluation Metrics (Real Price Values):")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # --- Plotting ---
    plt.figure(figsize=(14, 6))
    
    # Plot a specific segment for better visibility (e.g., first 500 points)
    plot_len = min(500, len(true_real_prices))
    
    plt.plot(true_real_prices[:plot_len], label='Real Stock Price (Close)', color='blue', linewidth=1.5)
    plt.plot(pred_real_prices[:plot_len], label='Predicted Price (iTransformer)', color='red', linestyle='--', linewidth=1.5)
    
    plt.title('Stock Price Forecasting: Real vs Predicted', fontsize=16)
    plt.xlabel('Time Steps (Days/Hours)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('stock_prediction_result.png', dpi=300)
    plt.show()
    print("Plot saved as 'stock_prediction_result.png'.")

if __name__ == "__main__":
    # 0. Fetch the data dynamically
    fetch_stock_data(TICKER, START_DATE, END_DATE, CSV_PATH)
    
    # 1. Train
    trained_model, test_loader, data_scaler = train_model()
    
    # 2. Evaluate, print metrics, and plot
    evaluate_and_plot(trained_model, test_loader, data_scaler)