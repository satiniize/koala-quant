import torch
from lstm_model import MultiTargetFinanceModel
import data_loader
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, num_epochs=200, device=torch.device("cpu")):
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

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
	# For demonstration, training on dummy data:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	num_features_ts = 5      # [Open, High, Low, Close, Volume]

	price_history = data_loader.get_ticker_data()
	normalized_history, scalers = data_loader.normalize_ticker_data(price_history)
	time_series_data, targets = data_loader.create_sequence_data(normalized_history)

	dataset = data_loader.FinanceDataset(time_series_data, targets)
	train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

	hidden_size_time_series = 64
	num_layers_time_series = 4

	model = MultiTargetFinanceModel(
		num_features_ts,
		hidden_size_time_series,
		num_layers_time_series,
	)
	model.to(device)
	train_model(model, train_loader, num_epochs=200, device=device)
