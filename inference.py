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
    device = next(model.parameters()).device

    # Initialize 3 separate windows for each path
    low_window = initial_window.copy()
    med_window = initial_window.copy()
    high_window = initial_window.copy()

    # Store predictions for each path
    low_predictions = []
    med_predictions = []
    high_predictions = []

    for i in range(num_days):
        model.eval()
        with torch.no_grad():
            # Predict low path
            input_low = torch.tensor(low_window, dtype=torch.float32).unsqueeze(0).to(device)
            preds_low = model(input_low, None)[0, :, 0].cpu().numpy()  # Use 0.1 quantile

            # Predict med path
            input_med = torch.tensor(med_window, dtype=torch.float32).unsqueeze(0).to(device)
            preds_med = model(input_med, None)[0, :, 1].cpu().numpy()  # Use 0.5 quantile

            # Predict high path
            input_high = torch.tensor(high_window, dtype=torch.float32).unsqueeze(0).to(device)
            preds_high = model(input_high, None)[0, :, 2].cpu().numpy()  # Use 0.9 quantile

        # Inverse transform predictions for each path
        # Low path
        preds_original_low = np.zeros_like(preds_low)
        preds_original_low[:4] = price_scaler.inverse_transform(preds_low[:4].reshape(-1, 1)).flatten()
        preds_original_low[4] = volume_scaler.inverse_transform(preds_low[4].reshape(-1, 1)).flatten()[0]

        # Med path
        preds_original_med = np.zeros_like(preds_med)
        preds_original_med[:4] = price_scaler.inverse_transform(preds_med[:4].reshape(-1, 1)).flatten()
        preds_original_med[4] = volume_scaler.inverse_transform(preds_med[4].reshape(-1, 1)).flatten()[0]

        # High path
        preds_original_high = np.zeros_like(preds_high)
        preds_original_high[:4] = price_scaler.inverse_transform(preds_high[:4].reshape(-1, 1)).flatten()
        preds_original_high[4] = volume_scaler.inverse_transform(preds_high[4].reshape(-1, 1)).flatten()[0]

        # Store predictions
        low_predictions.append(preds_original_low)
        med_predictions.append(preds_original_med)
        high_predictions.append(preds_original_high)

        # Update windows with their respective normalized predictions
        low_window = np.vstack([low_window[1:], preds_low])
        med_window = np.vstack([med_window[1:], preds_med])
        high_window = np.vstack([high_window[1:], preds_high])

    return low_predictions, med_predictions, high_predictions


# Compare predictions with actual data
def compare_predictions(price_history, predictions, window_size, offset=0):
    pessimistic, median, optimistic = predictions

    # Extract close prices from each path
    predicted_low = [pred[3] for pred in pessimistic]    # Close price from pessimistic path
    predicted_med = [pred[3] for pred in median]         # Close price from median path
    predicted_high = [pred[3] for pred in optimistic]    # Close price from optimistic path

    # Find correct index range
    n_predictions = len(predicted_med)
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
    plt.plot(prediction_x, predicted_med,
             label='Predicted Median Path',
             color='red',
             linestyle='--',
             linewidth=2)

    # Plot high and low paths
    plt.plot(prediction_x, predicted_high,
             label='Optimistic Path',
             color='green',
             linestyle=':',
             linewidth=1)

    plt.plot(prediction_x, predicted_low,
             label='Pessimistic Path',
             color='orange',
             linestyle=':',
             linewidth=1)

    # Fill between high and low predictions
    plt.fill_between(prediction_x, predicted_low, predicted_high,
                     color='red', alpha=0.2, label="Prediction Range")

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

    offset = 100
    ticker_symbol = config["prediction"]["ticker_symbol"]
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
    # future_predictions = predict_future_values(
    #     model,
    #     initial_window,
    #     scalers,
    #     num_days=num_future_days,
    #     quantiles=quantiles
    # )
    pessimistic, median, optimistic = predict_future_values(
        model,
        initial_window,
        scalers,
        num_days=num_future_days,
        quantiles=quantiles
    )

    # Print predictions
    # print("\nPredictions:")
    # for i, preds in enumerate(future_predictions, start=1):
    #     print(f"  Open (10%-50%-90%):   {float(preds[0][0]):,.2f} - {float(preds[1][0]):,.2f} - {float(preds[2][0]):,.2f}")
    #     print(f"  High (10%-50%-90%):   {float(preds[0][1]):,.2f} - {float(preds[1][1]):,.2f} - {float(preds[2][1]):,.2f}")
    #     print(f"  Low (10%-50%-90%):    {float(preds[0][2]):,.2f} - {float(preds[1][2]):,.2f} - {float(preds[2][2]):,.2f}")
    #     print(f"  Close (10%-50%-90%):  {float(preds[0][3]):,.2f} - {float(preds[1][3]):,.2f} - {float(preds[2][3]):,.2f}")
    #     print(f"  Volume (10%-50%-90%): {int(preds[0][4])} - {int(preds[1][4])} - {int(preds[2][4])}")

    # Plot predictions
    compare_predictions(price_history, (pessimistic, median, optimistic), window_size, offset)
