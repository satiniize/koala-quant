try:
	import tomllib
except:
	import tomli as tomllib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformer_model2 import create_transformer
import data_loader

# TODO: Add validation, add checkpoints
def train_model(model, train_loader, num_epochs=200, device=torch.device("cpu"), learning_rate=0.001, lr_scheduler_step=20, lr_scheduler_gamma=0.9):
    # Change to CrossEntropyLoss since we're predicting over vocabulary
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_ts, y in train_loader:
            x_ts, y = x_ts.to(device), y.to(device)
            optimizer.zero_grad()

            # outputs shape: [batch_size, seq_len, vocab_size]
            outputs = model(x_ts)

            # Reshape outputs and targets for CrossEntropyLoss
            # outputs: [batch_size * seq_len, vocab_size]
            # y: [batch_size]
            outputs = outputs[:, -1, :]  # Get last sequence predictions [batch_size, vocab_size]

            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

    save_path = "multi_target_finance_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
	with open('model_hyperparam.toml', 'rb') as f:
		config = tomllib.load(f)

	# Model parameters for transformer
	vocab_size = config['model']['vocab_size']
	d_model = config['model']['d_model']
	num_heads = config['model']['num_heads']
	num_layers = config['model']['num_layers']
	d_ff = config['model']['d_ff']
	max_seq_length = config['model']['max_seq_length']
	dropout = config['model']['dropout']

	# Training
	num_epochs = config['training']['num_epochs']
	batch_size = config['training']['batch_size']
	learning_rate = config['training']['learning_rate']
	lr_scheduler_step = config['training']['lr_scheduler_step']
	lr_scheduler_gamma = config['training']['lr_scheduler_gamma']

	# Data
	sequence_length = config['data']['sequence_length']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Create large finance dataset
	all_data = []
	for ticker_info in config['data']['training_tickers']:
		price_history = data_loader.get_ticker_data(ticker_info["ticker"], period=ticker_info["period"])
		normalized_history, _ = data_loader.normalize_ticker_data(price_history)
		tokenized_history = data_loader.tokenize_ticker_data(normalized_history, vocab_size)
		all_data.append(tokenized_history)

	combined_time_series = np.concatenate(all_data, axis=0)

	dataset = data_loader.FinanceDataset(combined_time_series, sequence_length)
	train_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
	)

	# Define and train model
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
	train_model(
		model,
		train_loader,
		num_epochs=num_epochs,
		device=device,
		learning_rate=learning_rate,
		lr_scheduler_step=lr_scheduler_step,
		lr_scheduler_gamma=lr_scheduler_gamma,
	)
