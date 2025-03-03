import torch
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from model import MultiTargetFinanceModel, FinanceDataset

def load_data():
    ts_data = np.load("ts_data.npy")
    tab_data = np.load("tab_data.npy")
    targets = np.load("targets.npy")
    return ts_data, tab_data, targets

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_ts, x_tab, y in data_loader:
            x_ts, x_tab = x_ts.to(device), x_tab.to(device)
            outputs = model(x_ts, x_tab)  # expected shape: (batch, 5)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y.cpu().numpy())
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    rmse = math.sqrt(mse)
    print(f"Evaluation RMSE (normalized): {rmse:.4f}")
    return rmse

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_data, tab_data, targets = load_data()

    num_samples = ts_data.shape[0]
    split_idx = int(0.8 * num_samples)
    ts_test = ts_data[split_idx:]
    tab_test = tab_data[split_idx:]
    targets_test = targets[split_idx:]

    test_dataset = FinanceDataset(ts_test, tab_test, targets_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_features_ts = ts_test.shape[2]     # typically 5
    num_features_tab = tab_test.shape[1]     # e.g., 10
    hidden_size_ts = 64
    num_layers_ts = 2
    hidden_size_tab = 32

    # Instantiate the multi-target model (which outputs 5 predictions)
    model = MultiTargetFinanceModel(num_features_ts, hidden_size_ts, num_layers_ts,
                                    num_features_tab, hidden_size_tab)
    model.to(device)

    # Load the trained multi-target model weights
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("Multi-target model loaded.")

    evaluate_model(model, test_loader, device)
