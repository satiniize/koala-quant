import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fetch_data(ticker_symbol="ITMG.JK", period="2y", window_size=20):
    ticker = yf.Ticker(ticker_symbol)

    # Download historical price data (using a longer period)
    hist = ticker.history(period=period)
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    hist.reset_index(inplace=True)

    ts_data = []
    targets = []
    for i in range(len(hist) - window_size - 1):
        window = hist.iloc[i:i+window_size]
        window_array = window[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
        ts_data.append(window_array)
        # Modified: extract multiple target variables instead of just 'Close'
        target = hist.iloc[i+window_size][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
        targets.append(target)

    ts_data = np.array(ts_data)
    targets = np.array(targets)  # Now targets have shape: (num_samples, 5)

    # Normalize time series data across the feature dimension
    num_samples, seq_len, num_features = ts_data.shape
    ts_data = ts_data.reshape(-1, num_features)
    ts_scaler = StandardScaler()
    ts_data_scaled = ts_scaler.fit_transform(ts_data).reshape(num_samples, seq_len, num_features)

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
    np.save("ts_data.npy", ts_data_scaled)
    np.save("tab_data.npy", tab_data_scaled)
    np.save("targets.npy", targets_scaled)
    print("Data saved to ts_data.npy, tab_data.npy, and targets.npy")
    return ts_data_scaled, tab_data_scaled, targets_scaled

if __name__ == "__main__":
    ts_data, tab_data, targets = fetch_data()
