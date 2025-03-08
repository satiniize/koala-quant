try:
    import tomllib
except:
    import tomli as tomllib

import torch
import numpy as np
from transformer_model2 import create_transformer
import data_loader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def predict_next_token(model, input_sequence, device):
    model.eval()
    with torch.no_grad():
        # Forward pass through the model
        output = model(input_sequence.to(device))
        # Get the last prediction
        last_prediction = output[:, -1, :]  # [batch_size, vocab_size]
        # Get the most likely token
        predicted_token = torch.argmax(last_prediction, dim=-1)
        return predicted_token.item()

def predict_future_values(model, initial_tokens, vocab_size, num_predictions=5, device=torch.device("cpu")):
    current_sequence = initial_tokens.clone()
    predictions = []

    for _ in range(num_predictions):
        # Prepare input sequence
        input_sequence = current_sequence.unsqueeze(0)  # Add batch dimension

        # Get prediction
        predicted_token = predict_next_token(model, input_sequence, device)
        predictions.append(predicted_token)

        # Update sequence for next prediction
        current_sequence = torch.cat([current_sequence[1:], torch.tensor([predicted_token])])

    return predictions

def detokenize_predictions(predictions, vocab_size):
    bins = np.linspace(-10.0, 10.0, vocab_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    normalized_values = [bin_centers[token-1] for token in predictions]
    return normalized_values

def compare_predictions(price_data, future_predictions, sequence_length, scaler, offset=0):
    predicted_prices = scaler.inverse_transform(np.array(future_predictions))

    n_predictions = len(predicted_prices)
    n = len(price_data) - (sequence_length + n_predictions) - offset

    historical_prices = price_data[n:len(price_data)-offset]

    historical_x = range(sequence_length + n_predictions)
    prediction_x = range(sequence_length, sequence_length + n_predictions)

    plt.figure(figsize=(15, 8))
    plt.plot(historical_x, historical_prices, label='Historical Prices', color='blue', linewidth=2)
    plt.plot(prediction_x, predicted_prices, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
    plt.scatter(historical_x, historical_prices, color='blue', s=50)
    plt.scatter(prediction_x, predicted_prices, color='red', s=50)

    plt.title(f'Historical and Predicted Prices', fontsize=14, pad=20)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with open('model_hyperparam.toml', 'rb') as f:
        config = tomllib.load(f)

    # Load configuration
    vocab_size = config['model']['vocab_size']
    d_model = config['model']['d_model']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    d_ff = config['model']['d_ff']
    max_seq_length = config['model']['max_seq_length']
    dropout = config['model']['dropout']
    sequence_length = config['data']['sequence_length']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create and load model
    model = create_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    model.to(device)
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("\nModel loaded successfully.")

    # Get and prepare data
    ticker_symbol = "BMRI.JK"
    price_history = data_loader.get_ticker_data(ticker_symbol)
    normalized_history, price_scaler = data_loader.normalize_ticker_data(price_history)
    tokenized_history = data_loader.tokenize_ticker_data(normalized_history, vocab_size)

    # Get initial sequence for prediction
    initial_sequence = torch.tensor(tokenized_history[-sequence_length:], dtype=torch.long)

    # Make predictions
    num_predictions = config['prediction']['future_days']
    predictions = predict_future_values(
        model,
        initial_sequence,
        vocab_size,
        num_predictions=num_predictions,
        device=device
    )

    # Convert predictions back to normalized values
    normalized_predictions = detokenize_predictions(predictions, vocab_size)

    # Print predictions
    print("\nPredicted values (normalized):")
    for i, pred in enumerate(normalized_predictions, 1):
        print(f"Day {i}: {pred:.4f}")

    # Plot predictions
    compare_predictions(price_history, normalized_predictions, sequence_length, price_scaler)
