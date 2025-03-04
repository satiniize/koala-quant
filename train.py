try:
    import tomllib
except:
    import tomli as tomllib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lstm_model import MultiTargetFinanceModel
import data_loader

# TODO: Add validation, add checkpoints
def train_model(model, train_loader, num_epochs=200, device=torch.device("cpu"), learning_rate=0.001, lr_scheduler_step=20, lr_scheduler_gamma=0.9):
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

	model.train()
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		for x_ts, y in train_loader:
			x_ts, y = x_ts.to(device), y.to(device)
			optimizer.zero_grad()
			outputs = model(x_ts, None)
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

	# Model
	input_size_time_series = config['model']['input_size_time_series']      # [Open, High, Low, Close, Volume]
	hidden_size_time_series = config['model']['hidden_size_time_series']
	num_layers_time_series = config['model']['num_layers_time_series']
	dropout = config['model']['dropout']

	# Training
	num_epochs = config['training']['num_epochs']
	batch_size = config['training']['batch_size']
	learning_rate = config['training']['learning_rate']
	lr_scheduler_step = config['training']['lr_scheduler_step']
	lr_scheduler_gamma = config['training']['lr_scheduler_gamma']

	# Data
	training_tickers = config['data']['training_tickers']
	sequence_length = config['data']['sequence_length']

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Create large finance dataset
	all_time_series_data = []
	all_targets = []

	for ticker_info in training_tickers:
		price_history = data_loader.get_ticker_data(ticker_info["ticker"], period=ticker_info["period"])
		normalized_history, scalers = data_loader.normalize_ticker_data(price_history)
		time_series_data, targets = data_loader.create_sequence_data(normalized_history, sequence_length)
		all_time_series_data.append(time_series_data)
		all_targets.append(targets)

	combined_time_series = np.concatenate(all_time_series_data, axis=0)
	combined_targets = np.concatenate(all_targets, axis=0)

	dataset = data_loader.FinanceDataset(combined_time_series, combined_targets)
	train_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
	)

	# Define and train model
	model = MultiTargetFinanceModel(
		input_size_time_series,
		hidden_size_time_series,
		num_layers_time_series,
		dropout
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
