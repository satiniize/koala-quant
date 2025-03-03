# Core Data Libraries
import numpy as np
import yfinance as yf
from torch.utils.data import Dataset, DataLoader

# Data Processing
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Utilities
from pprint import pprint

def get_ticker_data(ticker_symbol: str = "^JKSE", period: str = "2y"):
	ticker = yf.Ticker(ticker_symbol)
	history = ticker.history(period=period)
	history = history[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
	history.reset_index(inplace=True)
	return history # Pandas DataFrame

def normalize_ticker_data(data: pd.DataFrame):
	scaler = StandardScaler()
	normalized_data = data.copy()
	normalized_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
	return normalized_data, scaler

def create_sequence_data(history: pd.DataFrame, window_size=30):
	time_series_data 	= []
	targets 			= []
	for i in range(len(history) - window_size - 1):
		window_array 	= history.iloc[i:i+window_size][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
		time_series_data.append(window_array)
		target 			= history.iloc[i+window_size][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
		targets.append(target)

	time_series_data 	= np.array(time_series_data)
	targets 			= np.array(targets)  # Now targets have shape: (num_samples, 5)

	# Normalize time series data across the feature dimension
	print(time_series_data.shape)
	num_samples, seq_len, num_features = time_series_data.shape

	# Extract tabular features
	info = yf.Ticker(history.index[0].strftime('%Y-%m-%d')).info
	numerical_keys = ['marketCap', 'beta', 'dividendYield', 'trailingPE',
					'forwardPE', 'bookValue', 'priceToBook', 'returnOnEquity',
					'debtToEquity', 'freeCashflow']
	tab_vector = []
	for key in numerical_keys:
		try:
			tab_vector.append(float(info.get(key, 0)))
		except:
			tab_vector.append(0.0)

	tab_vector = np.array(tab_vector).reshape(1, -1)
	tab_data = np.repeat(tab_vector, num_samples, axis=0)

	return time_series_data, tab_data, targets

# Custom dataset for multi-target prediction
class FinanceDataset(Dataset):
	def __init__(self, ts_data, tab_data, targets):
		self.ts_data = torch.from_numpy(ts_data).float()
		self.tab_data = torch.from_numpy(tab_data).float()
		self.targets = torch.from_numpy(targets).float()

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, idx):
		return self.ts_data[idx], self.tab_data[idx], self.targets[idx]

def get_recent_data(ticker_symbol="ITMG.JK", period="2y"):
	ticker = yf.Ticker(ticker_symbol)
	hist = ticker.history(period=period)
	hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
	hist.reset_index(inplace=True)
	return hist

if __name__ == "__main__":
	ticker_data = get_ticker_data()
	normalized_ticker_data, scaler = normalize_ticker_data(ticker_data)
	pprint(ticker_data)
	# time_series_data, tab_data, targets = fetch_data()
	# print(time_series_data.shape)
