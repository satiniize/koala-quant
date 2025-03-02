import yfinance as yf
dat = yf.Ticker("ITMG.JK")

# Financial data relevant for machine learning models
print("FINANCIALLY RELEVANT DATA FOR PREDICTIVE MODELING:")

# Historical price data - highly relevant for time series prediction models
print("\nHistorical Price Data (1 month):")
historical_data = dat.history(period='1mo')
print(historical_data)
print("ML Relevance: Critical for time series prediction, trend analysis, and return forecasting")

# Quarterly financial statements - relevant for fundamental analysis models
print("\nQuarterly Income Statement:")
quarterly_financials = dat.quarterly_income_stmt
print(quarterly_financials)
print("ML Relevance: Important for fundamental analysis models predicting earnings or financial health")

# Analyst price targets - relevant for consensus prediction models
print("\nAnalyst Price Targets:")
analyst_targets = dat.analyst_price_targets
print(analyst_targets)
print("ML Relevance: Useful as a feature for price prediction models")

# Options data - relevant for volatility and sentiment models
print("\nOptions Chain:")
if len(dat.options) > 0:
    options_data = dat.option_chain(dat.options[0]).calls
    print(options_data)
    print("ML Relevance: Valuable for implied volatility prediction and sentiment analysis")

# Calendar data - relevant for event-based prediction models
print("\nCalendar (Earnings dates, etc.):")
calendar_data = dat.calendar
print(calendar_data)
print("ML Relevance: Useful for event-based trading strategies and earnings surprise models")

# Multiple tickers for comparative analysis
print("\nMULTIPLE TICKERS DATA (Comparative Analysis):")
market_data = yf.download(['MSFT', 'AAPL', 'GOOG'], period='1mo')
print(market_data)
print("ML Relevance: Essential for cross-sectional analysis, relative valuation models, and sector performance")

# ETF data for market benchmark and sector analysis
print("\nSPY ETF DATA (Market Benchmark):")
spy = yf.Ticker('SPY').funds_data
print("Top Holdings:")
print(spy.top_holdings)
print("ML Relevance: Useful for market correlation models and sector rotation strategies")

# Key company info - selectively useful for feature engineering
print("\nSelected Company Info:")
info = dat.info
relevant_keys = ['sector', 'industry', 'marketCap', 'beta', 'dividendYield',
                'trailingPE', 'forwardPE', 'bookValue', 'priceToBook',
                'returnOnEquity', 'debtToEquity', 'freeCashflow']
for key in relevant_keys:
    if key in info:
        print(f"{key}: {info[key]}")
print("ML Relevance: Useful for feature engineering in fundamental analysis models")
