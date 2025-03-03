import torch
from lstm_model import MultiTargetFinanceModel
import data_loader

if __name__ == "__main__":
    # For demonstration, training on dummy data:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 1000       # number of samples
    seq_len = 20             # time window length
    num_features_ts = 5      # [Open, High, Low, Close, Volume]
    num_features_tab = 10    # fundamental features
    num_targets = 5          # predicting [Open, High, Low, Close, Volume]

    time_series_data, tab_data, targets = data_loader.fetch_data()

    dataset = FinanceDataset(time_series_data, tab_data, targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    hidden_size_time_series = 64
    num_layers_time_series = 2
    hidden_size_tab = 32

    model = MultiTargetFinanceModel(num_features_ts, hidden_size_time_series, num_layers_time_series,
                                    num_features_tab, hidden_size_tab)
    model.to(device)
    train_model(model, train_loader, num_epochs=500, device=device)
