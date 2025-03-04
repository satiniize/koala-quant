import numpy as np
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pprint import pprint
import tenacity

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(5))
def get_ticker_data(ticker_symbol: str = "^JKSE", period: str = "2y"):
	stock_ticker = yf.Ticker(ticker_symbol)
	price_history = stock_ticker.history(period=period)
	price_history = price_history[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
	price_history.reset_index(inplace=True)
	return price_history

def normalize_ticker_data(data: pd.DataFrame):
	normalized_data = data.copy()
	price_columns = ['Open', 'High', 'Low', 'Close']

	# Normalize OHLC together across both time and feature dimensions
	price_data = data[price_columns].values
	price_scaler = StandardScaler()
	price_scaler.fit(price_data.reshape(-1, 1))

	for i, col in enumerate(price_columns):
		normalized_data[col] = price_scaler.transform(price_data[:, i].reshape(-1, 1))

	# Normalize volume separately
	volume_data = data[['Volume']].values
	volume_scaler = StandardScaler()
	normalized_data['Volume'] = volume_scaler.fit_transform(volume_data)

	return normalized_data, (price_scaler, volume_scaler) #TODO: Double check if this is still open high low close volume

def normalized_per_cat(data: pd.DataFrame):
    normalized = data.copy()
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    scalers = {}

    for col in price_cols:
        scaler = StandardScaler()
        normalized[col] = scaler.fit_transform(data[[col]])
        scalers[col] = scaler

    return normalized, scalers

def create_sequence_data(history: pd.DataFrame, window_size=30):
	time_series_data = []
	targets = []
	for i in range(len(history) - window_size - 1):
		window_array = history.iloc[i:i+window_size][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
		time_series_data.append(window_array)
		target = history.iloc[i+window_size][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
		targets.append(target)

	time_series_data = np.array(time_series_data)
	targets = np.array(targets)  # Now targets have shape: (num_samples, 5)

	return time_series_data, targets

class FinanceDataset(Dataset):
	def __init__(self, ts_data, targets):
		self.ts_data = torch.from_numpy(ts_data).float()
		self.targets = torch.from_numpy(targets).float()

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, idx):
		return self.ts_data[idx], self.targets[idx]

if __name__ == "__main__":
	# price_history, fundamental_metrics = get_ticker_data()
	price_history = get_ticker_data()
	normalized_history, scalers = normalized_per_cat(price_history)
	time_series_data, targets = create_sequence_data(normalized_history)
	pprint(targets[0])
