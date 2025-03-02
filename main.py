import yfinance as yf
from datetime import datetime, timedelta

dat = yf.Ticker("^JKSE")
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

hist = dat.history(start=start_date, end=end_date)
print(hist['Close'])
