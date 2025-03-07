import numpy as np
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pprint import pprint
import tenacity

# 1. ln(x)
# 2. minus by gradient/trend ie y=bx
# 3. divide by mean(alternatively median?)

class FinanceScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scale_price	= 0.0
        self.slope_price	= 0.0
        self.epsilon		= 1e-10

    def fit(self, X, y=None):
        """Compute the scaling parameters"""
        x = np.arange(len(X))
        log_values = np.log(X)
        self.slope_price = np.polyfit(x, log_values, 1)[0]

        trend = self.slope_price * x
        detrended = log_values - trend
        self.scale_price = np.mean(np.abs(detrended))

        return self

    def transform(self, X):
        """Apply the transformation"""
        result = X.copy()

        x = np.arange(len(X))
        log_values = np.log(X)

        trend = self.slope_price * x
        detrended = log_values - trend

        normalized = detrended / self.scale_price

        result = normalized

        return result

    def inverse_transform(self, X):
        """Reverse the transformation"""
        result = X.copy()

        x = np.arange(len(X))

        unscaled = X * self.scale_price

        trend = self.slope_price * x
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
	return price_history

def normalize_ticker_data(data: pd.DataFrame):
	normalized_data = data.copy()
	price_data = data['Open'].values.reshape(-1, 1)
	price_scaler = FinanceScaler()
	price_scaler.fit(price_data)
	normalized_data['Open'] = price_scaler.transform(price_data)
	return normalized_data, price_scaler

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
		window_array = history.iloc[i:i+window_size][['Open']].values.astype(float)
		time_series_data.append(window_array)
		target = history.iloc[i+window_size][['Open']].values.astype(float)
		targets.append(target)

	time_series_data = np.array(time_series_data)
	targets = np.array(targets)  # Now targets have shape: (num_samples, 1)

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
