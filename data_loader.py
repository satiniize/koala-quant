try:
	import tomllib
except:
	import tomli as tomllib
import numpy as np
import pandas as pd
import tenacity
import torch
import math
import yfinance as yf
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
# 1. ln(x)
# 2. minus by gradient/trend ie y=bx
# 3. divide by mean(alternatively median?)
# 4. subtract by 1

class FinanceScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = 0.0
        self.slope = 0.0
        self.stddev = 0.0
        # self.epsilon = 1e-10

    def fit(self, X, y=None):
        """Compute the scaling parameters"""
        x = np.arange(len(X))
        log_values = np.log(X)
        self.slope = np.polyfit(x, log_values, 1)[0]

        trend = self.slope * x
        detrended = log_values - trend
        self.mean = np.mean(np.abs(detrended))
        self.stddev = np.std(detrended)
        return self

    def transform(self, X):
        """Apply the transformation"""
        result = X.copy()

        x = np.arange(len(X))
        log_values = np.log(X)

        trend = self.slope * x
        detrended = log_values - trend
        zero_mean = detrended - self.mean
        normalized = zero_mean / self.stddev
        result = normalized

        return result

    def inverse_transform(self, X):
        """Reverse the transformation"""
        result = X.copy()

        x = np.arange(len(X))

        unscaled = (X * self.stddev) + self.mean

        trend = self.slope * x
        with_trend = unscaled + trend

        original = np.exp(with_trend)

        result = original

        return result

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(5))
def get_ticker_data(ticker_symbol: str = "^JKSE", period: str = "2y", labels: str = ["Close"]):
	stock_ticker = yf.Ticker(ticker_symbol)
	price_history = stock_ticker.history(period=period)
	price_history = price_history[labels].dropna()
	price_history.reset_index(inplace=True)
	return price_history['Close'].to_numpy()

def normalize_ticker_data(data: np.ndarray):
	normalized_data = data.copy()
	price_data = data
	price_scaler = FinanceScaler()
	price_scaler.fit(price_data)
	normalized_data = price_scaler.transform(price_data)
	return normalized_data, price_scaler

def tokenize_ticker_data(data: np.ndarray, n_bins):
	tokenized_data = data.copy()
	bins = np.linspace(-10.0, 10.0, n_bins)
	tokenized_data = np.digitize(data, bins)
	return tokenized_data

def detokenize_ticker_data(tokenized_data: np.ndarray, n_bins):
	detokenized_data = tokenized_data.copy()
	bin_size = 10/n_bins
	detokenized_data['Close'] = tokenized_data['Close'] * bin_size
	return detokenized_data

def normalized_per_cat(data: pd.DataFrame):
    normalized = data.copy()
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    scalers = {}

    for col in price_cols:
        scaler = StandardScaler()
        normalized[col] = scaler.fit_transform(data[[col]])
        scalers[col] = scaler

    return normalized, scalers

# def create_sequence_data(history: np.ndarray, window_size=30):
# 	time_series_data = []
# 	targets = []
# 	for i in range(len(history) - window_size - 1):
# 		# Include all features you want to use
# 		window_array = history.iloc[i:i+window_size].values.astype(float)
# 		time_series_data.append(window_array)
# 		target = history.iloc[i+window_size]['Close']
# 		targets.append(target)

# 	return np.array(time_series_data), np.array(targets)

class FinanceDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.sequence_length = sequence_length
        self.data = torch.from_numpy(data).long()  # Convert to Long tensor for embedding

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return x, y

if __name__ == "__main__":
	with open('model_hyperparam.toml', 'rb') as f:
		config = tomllib.load(f)
	bins = config['model']['vocabulary_size']
	price_history = get_ticker_data()
	normalized_history, scalers = normalize_ticker_data(price_history)
	tokenized_time_series_data = tokenize_ticker_data(normalized_history, bins)

	print(tokenized_time_series_data)

	plt.figure(figsize=(10, 6))
	plt.plot(range(len(tokenized_time_series_data)), tokenized_time_series_data, label='Tokenized Data')
	plt.xlabel('Index')
	plt.ylabel('Token Value')
	plt.title('Tokenized Time Series Data')
	plt.legend()
	plt.grid(True)
	plt.show()
