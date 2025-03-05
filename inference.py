try:
    import tomllib
except:
    import tomli as tomllib

import torch
import numpy as np
from lstm_model import MultiTargetFinanceModel
import data_loader
import matplotlib.pyplot as plt
import pandas as pd

def get_initial_window(normalized_history, window_size=30):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    window = normalized_history.iloc[-window_size:][features].values.astype(float)
    return window

def predict_future_values(model, initial_window, scalers, num_days=5, quantiles=[0.1, 0.5, 0.9]):
    price_scaler, volume_scaler = scalers
    predictions = []
    current_window = initial_window.copy()
    device = next(model.parameters()).device

    for i in range(num_days):
        input_time_series = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            preds_norm = model(input_time_series, None)  # Shape: (1, 5, n_quantiles)

        # Extract quantiles
        lower_q = preds_norm[0, :, 0].cpu().numpy()  # Lower quantile (e.g., 10%)
        median_q = preds_norm[0, :, 1].cpu().numpy()  # Median (50%)
        upper_q = preds_norm[0, :, 2].cpu().numpy()  # Upper quantile (e.g., 90%)

        # Inverse transform predictions
        preds_original_lower = np.zeros_like(lower_q)
        preds_original_median = np.zeros_like(median_q)
        preds_original_upper = np.zeros_like(upper_q)

        # Inverse transform OHLC (first 4 values)
        preds_original_lower[:4] = price_scaler.inverse_transform(lower_q[:4].reshape(-1, 1)).flatten()
        preds_original_median[:4] = price_scaler.inverse_transform(median_q[:4].reshape(-1, 1)).flatten()
        preds_original_upper[:4] = price_scaler.inverse_transform(upper_q[:4].reshape(-1, 1)).flatten()

        # Inverse transform Volume (last value)
        preds_original_lower[4] = volume_scaler.inverse_transform(lower_q[4].reshape(-1, 1)).flatten()[0]
        preds_original_median[4] = volume_scaler.inverse_transform(median_q[4].reshape(-1, 1)).flatten()[0]
        preds_original_upper[4] = volume_scaler.inverse_transform(upper_q[4].reshape(-1, 1)).flatten()[0]

        predictions.append((preds_original_lower, preds_original_median, preds_original_upper))

        # Update window with normalized median predictions for next iteration
        current_window = np.vstack([current_window[1:], median_q])

    return predictions


# Compare predictions with actual data
def compare_predictions(price_history, future_predictions, window_size, offset=0):
    # Extract quartiles from predictions
    predicted_q25 = [pred[0][3] for pred in future_predictions]  # 25th percentile (Lower)
    predicted_q50 = [pred[1][3] for pred in future_predictions]  # Median
    predicted_q75 = [pred[2][3] for pred in future_predictions]  # 75th percentile (Upper)

    # Find correct index range
    n_predictions = len(predicted_q50)
    n = len(price_history) - (window_size + n_predictions) - offset  # Data cutoff

    historical_close = price_history['Close'].values
    historical_close = historical_close[n:len(historical_close)-offset]  # Adjust for offset

    # Create x-axis values
    historical_x = range(window_size + n_predictions)
    prediction_x = range(window_size, window_size + n_predictions)

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot historical data
    plt.plot(historical_x, historical_close,
             label='Historical Close Prices',
             color='blue',
             linewidth=2)

    # Plot median predictions
    plt.plot(prediction_x, predicted_q50,
             label='Predicted Median Close Prices',
             color='red',
             linestyle='--',
             linewidth=2)

    # Plot quartile range as shaded area
    plt.fill_between(prediction_x, predicted_q25, predicted_q75,
                     color='red', alpha=0.2, label="Interquartile Range (25%-75%)")

    # Add labels and title
    plt.title(f'Historical and Predicted Close Prices for {ticker_symbol}',
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
    with open('model_hyperparam.toml', 'rb') as f:
        config = tomllib.load(f)

    offset = 200
    ticker_symbol = "ASII.JK"
    window_size = config["data"]["sequence_length"]
    num_future_days = config["prediction"]["future_days"]

    # Get data and normalize it
    price_history = data_loader.get_ticker_data(ticker_symbol)
    normalized_history, scalers = data_loader.normalize_ticker_data(price_history)
    partial_normalized_history = normalized_history[:-(num_future_days+offset)]

    # Get initial window from normalized data
    initial_window = get_initial_window(partial_normalized_history, window_size=window_size)

    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define quantiles
    quantiles = [0.1, 0.5, 0.9]  # 10%, 50% (median), 90%

    input_size_time_series = config['model']['input_size_time_series']
    hidden_size_time_series = config['model']['hidden_size_time_series']
    num_layers_time_series = config['model']['num_layers_time_series']
    dropout = config['model']['dropout']

    # Load the model
    model = MultiTargetFinanceModel(
        input_size_time_series,
        hidden_size_time_series,
        num_layers_time_series,
        dropout,
        n_quantiles=len(quantiles)  # Add number of quantiles
    )
    model.to(device)
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("\nModel loaded successfully.")

    # Make predictions with quantiles
    future_predictions = predict_future_values(
        model,
        initial_window,
        scalers,
        num_days=num_future_days,
        quantiles=quantiles
    )

    # Print predictions
    print("\nPredictions:")
    for i, preds in enumerate(future_predictions, start=1):
        print(f"  Open (10%-50%-90%):   {float(preds[0][0]):,.2f} - {float(preds[1][0]):,.2f} - {float(preds[2][0]):,.2f}")
        print(f"  High (10%-50%-90%):   {float(preds[0][1]):,.2f} - {float(preds[1][1]):,.2f} - {float(preds[2][1]):,.2f}")
        print(f"  Low (10%-50%-90%):    {float(preds[0][2]):,.2f} - {float(preds[1][2]):,.2f} - {float(preds[2][2]):,.2f}")
        print(f"  Close (10%-50%-90%):  {float(preds[0][3]):,.2f} - {float(preds[1][3]):,.2f} - {float(preds[2][3]):,.2f}")
        print(f"  Volume (10%-50%-90%): {int(preds[0][4])} - {int(preds[1][4])} - {int(preds[2][4])}")

    # Plot predictions
    compare_predictions(price_history, future_predictions, num_future_days, offset)
