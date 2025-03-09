try:
    import tomllib
except:
    import tomli as tomllib

import torch
import numpy as np
from transformer import create_transformer
import finance_data
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
    bin_centers = bins
    normalized_values = [bin_centers[token] for token in predictions]
    return normalized_values

def compare_predictions(price_history, future_predictions, window_size, scaler, offset):
	# predicted_prices = scaler.inverse_transform(np.array(future_predictions), len(price_history - window_size - len(future_predictions) - offset))
	# predicted_prices = scaler.inverse_transform(np.array(future_predictions), len(price_history - len(future_predictions) - offset))
    # predicted_prices = scaler.inverse_transform(np.array(future_predictions))

    n_predictions = len(predicted_prices)
    n = len(price_history) - (window_size + n_predictions) - offset

    historical_prices = price_history[n:len(price_history)-offset]

    historical_x = range(window_size + n_predictions)
    prediction_x = range(window_size, window_size + n_predictions)

    plt.figure(figsize=(15, 8))
    plt.plot(historical_x, historical_prices, label='Historical Prices', color='blue', linewidth=2)
    plt.plot(prediction_x, future_predictions, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
    plt.scatter(historical_x, historical_prices, color='blue', s=50)
    plt.scatter(prediction_x, future_predictions, color='red', s=50)

    plt.title('Historical and Predicted Prices', fontsize=14, pad=20)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with open('transformer.toml', 'rb') as f:
        config = tomllib.load(f)

    offset = 0

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
    model.load_state_dict(torch.load(config["training"]["best_model_path"], map_location=device))
    print("\nModel loaded successfully.")

    # Get and prepare data
    ticker_symbol = "BMRI.JK"
    price_history = finance_data.get_ticker_data(ticker_symbol)
    normalized_history, price_scaler = finance_data.normalize_ticker_data(price_history)
    tokenized_history = finance_data.tokenize_ticker_data(normalized_history, vocab_size)
    num_predictions = config['prediction']['future_days']
    partial_tokenized_history = tokenized_history[:-(num_predictions+offset)]
    initial_sequence = torch.tensor(partial_tokenized_history[-sequence_length:], dtype=torch.long)
    predictions = predict_future_values(
        model,
        initial_sequence,
        vocab_size,
        num_predictions=num_predictions,
        device=device
    )
    combined_tokens = np.concatenate([partial_tokenized_history, np.array(predictions)])
    normalized_predictions = detokenize_predictions(combined_tokens, vocab_size)

    # Convert predictions back to normalized values
    predicted_prices = price_scaler.inverse_transform(np.array(normalized_predictions))[-num_predictions:]

    # Print predictions
    print("\nPredicted values (normalized):")
    for i, pred in enumerate(normalized_predictions, 1):
        print(f"Day {i}: {pred:.4f}")

    # Plot predictions
    compare_predictions(price_history, predicted_prices, sequence_length, price_scaler, offset)
