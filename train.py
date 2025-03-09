"""Module for training a transformer model on financial data."""

try:
	import tomllib
except ImportError:
	import tomli as tomllib

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from transformer import create_transformer
import finance_data

def train_model(transformer, train_loader, val_loader, training_params, training_device):
	"""Train the transformer model.

	Args:
		transformer: The transformer model to train
		data_loader: DataLoader containing training data
		training_params: Dictionary containing training parameters
		training_device: Device to train on (CPU/GPU)
	"""
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(transformer.parameters(), lr=training_params['learning_rate'])
	scheduler = optim.lr_scheduler.StepLR(
		optimizer,
		step_size=training_params['lr_scheduler_step'],
		gamma=training_params['lr_scheduler_gamma']
	)

	best_val_loss = float('inf')

	for epoch in range(training_params['num_epochs']):
		transformer.train()
		train_loss = 0.0
		for x_ts, y in train_loader:
			x_ts, y = x_ts.to(training_device), y.to(training_device)
			optimizer.zero_grad()
			# outputs shape: [batch_size, seq_len, vocab_size]
			outputs = transformer(x_ts)

			outputs = outputs[:, -1, :]  # Get last sequence predictions [batch_size, vocab_size]

			loss = criterion(outputs, y)

			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		avg_train_loss = train_loss / len(train_loader)

		transformer.eval()
		val_loss = 0.0
		with torch.no_grad():
			for x_ts, y in val_loader:
				x_ts, y = x_ts.to(training_device), y.to(training_device)
				outputs = transformer(x_ts)
				outputs = outputs[:, -1, :]
				loss = criterion(outputs, y)
				val_loss += loss.item()

		avg_val_loss = val_loss / len(val_loader)

        # Save best model
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(transformer.state_dict(), training_params["best_model_path"])

		scheduler.step()
		print(
			f"Epoch [{epoch+1}/{training_params['num_epochs']}], "
			f"Train Loss: {avg_train_loss:.6f}, "
			f"Val Loss: {avg_val_loss:.6f}"
		)

	torch.save(transformer.state_dict(), training_params["final_model_path"])
	print(f"Best validation loss: {best_val_loss:.6f}")
	print("Training completed!")

if __name__ == "__main__":
	with open('transformer.toml', 'rb') as f:
		config = tomllib.load(f)

	# Model parameters for transformer
	vocab_size = config['model']['vocab_size']
	d_model = config['model']['d_model']
	num_heads = config['model']['num_heads']
	num_layers = config['model']['num_layers']
	d_ff = config['model']['d_ff']
	max_seq_length = config['model']['max_seq_length']
	dropout = config['model']['dropout']

	training_config = config['training']

	# Data
	sequence_length = config['data']['sequence_length']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Create large finance dataset
	train_data = []
	for ticker in config['data']['training_tickers']:
		price_history = finance_data.get_ticker_data(ticker, period=config['data']['period'])
		normalized_history, _ = finance_data.normalize_ticker_data(price_history)
		tokenized_history = finance_data.tokenize_ticker_data(normalized_history, vocab_size)
		train_data.append(tokenized_history)

	val_data = []
	for ticker in config['data']['validation_tickers']:
		price_history = finance_data.get_ticker_data(ticker, period=config['data']['period'])
		normalized_history, _ = finance_data.normalize_ticker_data(price_history)
		tokenized_history = finance_data.tokenize_ticker_data(normalized_history, vocab_size)
		val_data.append(tokenized_history)

	combined_train_series = np.concatenate(train_data, axis=0)
	combined_val_series = np.concatenate(val_data, axis=0)

	train_dataset = finance_data.FinanceDataset(combined_train_series, sequence_length)
	val_dataset = finance_data.FinanceDataset(combined_val_series, sequence_length)
	train_loader = DataLoader(
		train_dataset,
		batch_size=training_config['batch_size'],
		shuffle=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=training_config['batch_size'],
		shuffle=False,  # No need to shuffle validation data
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
		val_loader,
		training_config,
		device
	)
