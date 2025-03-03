import numpy as np
# import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from pprint import pprint

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
    num_samples, seq_len, num_features = time_series_data.shape
    time_series_data = time_series_data.reshape(-1, num_features)
    time_series_scaler = StandardScaler()
    time_series_data_scaled = time_series_scaler.fit_transform(time_series_data).reshape(num_samples, seq_len, num_features)

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

    # Repeat the tabular vector for each sample and normalize
    tab_data = np.repeat(tab_vector, num_samples, axis=0)
    tab_scaler = StandardScaler()
    tab_data_scaled = tab_scaler.fit_transform(tab_data)

    # Normalize targets (now with shape (num_samples, 5))
    target_scaler = StandardScaler()
    targets_scaled = target_scaler.fit_transform(targets)

    # Optionally, you can save these scalers for later inverse-transformation
    np.save("time_series_data.npy", time_series_data_scaled)
    np.save("tab_data.npy", tab_data_scaled)
    np.save("targets.npy", targets_scaled)
    print("Data saved to time_series_data.npy, tab_data.npy, and targets.npy")
    return time_series_data_scaled, tab_data_scaled, targets_scaled

if __name__ == "__main__":
    time_series_data, tab_data, targets = fetch_data()
    print(time_series_data[0])
