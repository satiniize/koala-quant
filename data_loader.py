import numpy as np
# import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def fetch_data(ticker_symbol="ITMG.JK", period="2y", window_size=30):
    ticker 				= yf.Ticker(ticker_symbol)

    history 			= ticker.history(period=period)
    history 			= history[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    history.reset_index(inplace=True)

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
    info = ticker.info
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


def get_recent_data(ticker_symbol="ITMG.JK", period="2y"):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    hist.reset_index(inplace=True)
    return hist

def convert_to_window():
	pass

def normalize():
	pass

if __name__ == "__main__":
    time_series_data, tab_data, targets = fetch_data()
    print(time_series_data.shape)
