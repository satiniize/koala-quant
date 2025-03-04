# inference.py
import torch
import numpy as np
from lstm_model import MultiTargetFinanceModel
import data_loader

def get_initial_window(normalized_history, window_size=30):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    window = normalized_history.iloc[-window_size:][features].values.astype(float)
    return window

def predict_future_values(model, initial_window, scalers, num_days=5):
    price_scaler, volume_scaler = scalers
    predictions = []
    current_window = initial_window.copy()
    device = next(model.parameters()).device

    for i in range(num_days):
        input_time_series = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            preds_norm = model(input_time_series, None)  # Shape: (1, 5)
            preds_norm = preds_norm.cpu().numpy()[0]

        # Inverse transform predictions
        preds_original = np.zeros_like(preds_norm)

        # Inverse transform OHLC (first 4 values)
        ohlc_values = preds_norm[:4].reshape(-1, 1)
        preds_original[:4] = price_scaler.inverse_transform(ohlc_values).flatten()

        # Inverse transform Volume (last value)
        volume_value = np.array([[preds_norm[4]]])  # Reshape to 2D array
        preds_original[4] = volume_scaler.inverse_transform(volume_value).item()

        predictions.append(preds_original)

        # Update window with normalized predictions for next iteration
        current_window = np.vstack([current_window[1:], preds_norm])

    return predictions

if __name__ == "__main__":
    ticker_symbol = "ITMG.JK"
    window_size = 30
    num_future_days = 5

    # Get data and normalize it
    price_history = data_loader.get_ticker_data(ticker_symbol)
    normalized_history, scalers = data_loader.normalize_ticker_data(price_history)

    # # Print some statistics about the original data
    # print("\nOriginal Data Statistics:")
    # print(price_history.describe())

    # Get initial window from normalized data
    initial_window = get_initial_window(normalized_history, window_size=window_size)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features_ts = 5
    hidden_size_ts = 64
    num_layers_ts = 4

    # Load the model
    model = MultiTargetFinanceModel(num_features_ts, hidden_size_ts, num_layers_ts)
    model.to(device)
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("\nModel loaded successfully.")

    # Make predictions
    future_predictions = predict_future_values(model, initial_window, scalers, num_days=num_future_days)

    # Print predictions
    print("\nPredictions:")
    for i, preds in enumerate(future_predictions, start=1):
        print(f"\nDay {i}:")
        print(f"  Open:   {preds[0]:,.2f}")
        print(f"  High:   {preds[1]:,.2f}")
        print(f"  Low:    {preds[2]:,.2f}")
        print(f"  Close:  {preds[3]:,.2f}")
        print(f"  Volume: {preds[4]:,.0f}")

    # Plot predictions
    plot_predictions(price_history, future_predictions, 30)
