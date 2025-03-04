import torch
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import data_loader
from lstm_model import MultiTargetFinanceModel

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x_ts, y in data_loader:
            x_ts = x_ts.to(device)
            y = y.to(device)

            outputs = model(x_ts, None)  # None for x_tab as it's not used in current model
            predictions.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Calculate MSE for each target separately
    mse_by_target = mean_squared_error(actuals, predictions, multioutput='raw_values')
    rmse_by_target = np.sqrt(mse_by_target)

    # Overall RMSE
    overall_mse = mean_squared_error(actuals, predictions)
    overall_rmse = math.sqrt(overall_mse)

    print("\nEvaluation Results (normalized):")
    print(f"Open  RMSE: {rmse_by_target[0]:.4f}")
    print(f"High  RMSE: {rmse_by_target[1]:.4f}")
    print(f"Low   RMSE: {rmse_by_target[2]:.4f}")
    print(f"Close RMSE: {rmse_by_target[3]:.4f}")
    print(f"Volume RMSE: {rmse_by_target[4]:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")

    return overall_rmse, rmse_by_target

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get test data
    ticker_symbol = "ITMG.JK"  # You can change this to any ticker you want to evaluate
    price_history = data_loader.get_ticker_data(ticker_symbol)
    normalized_history, scalers = data_loader.normalize_ticker_data(price_history)

    # Create sequences
    time_series_data, targets = data_loader.create_sequence_data(normalized_history)

    # Split into train/test (use last 20% for testing)
    split_idx = int(0.8 * len(time_series_data))
    ts_test = time_series_data[split_idx:]
    targets_test = targets[split_idx:]

    # Create test dataset and dataloader
    test_dataset = data_loader.FinanceDataset(ts_test, targets_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model parameters
    num_features_ts = 5  # [Open, High, Low, Close, Volume]
    hidden_size_ts = 64
    num_layers_ts = 4

    # Initialize model
    model = MultiTargetFinanceModel(num_features_ts, hidden_size_ts, num_layers_ts)
    model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("Model loaded successfully.")

    # Evaluate
    overall_rmse, target_rmse = evaluate_model(model, test_loader, device)
