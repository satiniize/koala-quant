import torch
# import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from lstm_model import MultiTargetFinanceModel

def prepare_scalers(hist):
    # Fit a scaler for the time series features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = hist[features].values.astype(float)
    ts_scaler = StandardScaler()
    ts_scaler.fit(X)

    # For multi-target prediction, we assume targets are the same features
    target_scaler = StandardScaler()
    target_scaler.fit(X)

    return ts_scaler, target_scaler

def get_initial_window(hist, window_size=20):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    window = hist.iloc[-window_size:][features].values.astype(float)
    # window_scaled = ts_scaler.transform(window)
    return window

def get_tab_vector(ticker_symbol="ITMG.JK"):
	# TODO: Move this to market_data
    ticker = yf.Ticker(ticker_symbol)
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
    return tab_vector

def predict_future_values(model, initial_window, tab_vector, num_days=5):
    predictions = []
    current_window = initial_window.copy()
    device = next(model.parameters()).device

    for i in range(num_days):
        input_time_series = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
        input_tab = torch.tensor(tab_vector, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            preds_norm = model(input_time_series, input_tab)  # Shape: (1, 5)
        preds_norm = preds_norm.cpu().numpy()[0]
        # Inverse-transform predictions to original scale
        # preds_original = target_scaler.inverse_transform(preds_norm.reshape(1, -1))[0]
        # predictions.append(preds_original)
        predictions.append(preds_norm)

        # Update window: assume new day's features are those predicted
        new_day_raw = preds_norm  # [Open, High, Low, Close, Volume]
        # new_day_scaled = ts_scaler.transform(new_day_raw.reshape(1, -1))[0]
        current_window = np.vstack([current_window[1:], new_day_raw])

    return predictions

if __name__ == "__main__":
    ticker_symbol = "ITMG.JK"
    window_size = 30
    num_future_days = 5

    # Fetch historical data and prepare the scalers and initial window
    hist = get_recent_data(ticker_symbol=ticker_symbol, period="2y")
    # ts_scaler, target_scaler = prepare_scalers(hist)
    initial_window = get_initial_window(hist, window_size=window_size)
    tab_vector = get_tab_vector(ticker_symbol=ticker_symbol)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features_ts = 5  # Open, High, Low, Close, Volume
    hidden_size_ts = 64
    num_layers_ts = 2
    num_features_tab = tab_vector.shape[1]
    hidden_size_tab = 32

    # Load the multi-target model
    model = MultiTargetFinanceModel(num_features_ts, hidden_size_ts, num_layers_ts,
                                    num_features_tab, hidden_size_tab)
    model.to(device)
    model.load_state_dict(torch.load("multi_target_finance_model.pth", map_location=device))
    print("Multi-target model loaded.")

    future_predictions = predict_future_values(model, initial_window, tab_vector, num_days=num_future_days)
    for i, preds in enumerate(future_predictions, start=1):
        print(f"Day {i} predictions: Open={preds[0]:,.2f}, High={preds[1]:,.2f}, "
              f"Low={preds[2]:,.2f}, Close={preds[3]:,.2f}, Volume={preds[4]:,.2f}")
