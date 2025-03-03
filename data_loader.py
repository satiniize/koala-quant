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
	stock_ticker = yf.Ticker(ticker_symbol)
	price_history = stock_ticker.history(period=period)
	price_history = price_history[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
	price_history.reset_index(inplace=True)

	# Calculate integrated features
	stock_info = stock_ticker.info
	fundamental_metrics = [
		'marketCap', 'beta', 'dividendYield', 'trailingPE',
		'forwardPE', 'bookValue', 'priceToBook', 'returnOnEquity',
		'debtToEquity', 'freeCashflow'
	]

	# Create tabular data vector
	fundamental_values = []
	for metric in fundamental_metrics:
		try:
			fundamental_values.append(float(stock_info.get(metric, 0)))
		except:
			fundamental_values.append(0.0)

	fundamental_data = np.array(fundamental_values)

	return price_history, fundamental_data # Return both history and tabular data

def normalize_ticker_data(data: pd.DataFrame):
	normalized_data = data.copy()
	price_columns = ['Open', 'High', 'Low', 'Close']

	# Normalize OHLC together across both time and feature dimensions
	price_scaler = StandardScaler()
	flattened_prices = data[price_columns].values.ravel()
	price_scaler.fit(flattened_prices.reshape(-1, 1))

	for col in price_columns:
		normalized_data[col] = price_scaler.transform(data[[col]])

	# Normalize volume separately
	volume_scaler = StandardScaler()
	normalized_data['Volume'] = volume_scaler.fit_transform(data[['Volume']])

	return normalized_data, (price_scaler, volume_scaler) #TODO: Double check if this is still open high low close volume

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

# def get_recent_data(ticker_symbol="ITMG.JK", period="2y"):
# 	ticker = yf.Ticker(ticker_symbol)
# 	hist = ticker.history(period=period)
# 	hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
# 	hist.reset_index(inplace=True)
# 	return hist

if __name__ == "__main__":
	price_history, fundamental_metrics = get_ticker_data()
	normalized_history, scalers = normalize_ticker_data(price_history)
	time_series_data, targets = create_sequence_data(normalized_history)
	pprint(targets[0])
