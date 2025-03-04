try:
    import tomllib
except:
    import tomli as tomllib

import torch
import numpy as np
from lstm_model import MultiTargetFinanceModel
import data_loader
import plot
import matplotlib.pyplot as plt
import pandas as pd

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

# Compare predictions with actual data
def compare_predictions(price_history, future_predictions, window_size, split):
    # Extract historical Open prices
    predicted_opens = [pred[0] for pred in future_predictions]

    n = int(len(price_history) * split) # Data cutoff
    n_predictions = len(predicted_opens)

    historical_opens = price_history['Open'].values[n-window_size:n+n_predictions]
    # Create x-axis values
    historical_x = range(window_size + n_predictions)
    prediction_x = range(window_size, window_size + n_predictions)

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot historical data
    plt.plot(historical_x, historical_opens,
             label='Historical Open Prices',
             color='blue',
             linewidth=2)

    # Plot predictions
    plt.plot(prediction_x, predicted_opens,
             label='Predicted Open Prices',
             color='red',
             linestyle='--',
             linewidth=2)

    # Add markers at data points
    plt.scatter(historical_x, historical_opens, color='blue', s=50)
    plt.scatter(prediction_x, predicted_opens, color='red', s=50)

    # Add labels and title
    plt.title(f'Historical and Predicted Open Prices for {ticker_symbol}',
              fontsize=14,
              pad=20)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Format y-axis with comma separator
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    ticker_symbol = "PTBA.JK"
    window_size = 30
    num_future_days = 64

    # Get data and normalize it
    price_history = data_loader.get_ticker_data(ticker_symbol)
    normalized_history, scalers = data_loader.normalize_ticker_data(price_history)
    split = 0.6
    partial_normalized_history = normalized_history[:int(len(normalized_history) * split)]

    # Get initial window from normalized data
    initial_window = get_initial_window(partial_normalized_history, window_size=window_size)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('model_hyperparam.toml', 'rb') as f:
        config = tomllib.load(f)

    input_size_time_series = config['model']['input_size_time_series']      # [Open, High, Low, Close, Volume]
    hidden_size_time_series = config['model']['hidden_size_time_series']
    num_layers_time_series = config['model']['num_layers_time_series']

    # Load the model
    model = MultiTargetFinanceModel(input_size_time_series, hidden_size_time_series, num_layers_time_series)
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
    compare_predictions(price_history, future_predictions, 30, split)
