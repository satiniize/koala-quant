import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(price_history, future_predictions, window_size):
    # Extract historical Open prices
    historical_opens = price_history['Open'].values[-window_size:]

    # Extract predicted Open prices
    predicted_opens = [pred[0] for pred in future_predictions]

    # Create x-axis values
    historical_x = range(window_size)  # 0 to window_size-1
    prediction_x = range(window_size, window_size + len(predicted_opens))  # window_size to window_size+predictions

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
