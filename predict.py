import yfinance as yf
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def fetch_data(ticker):
    df = yf.download(ticker, period="30d", interval="1d")
    df.dropna(inplace=True)
    # Normalize columns to lowercase to match training
    df.columns = df.columns.str.strip().str.lower()
    return df

def create_features(df):
    df['return'] = df['close'].pct_change()
    df['high_low'] = df['high'] - df['low']
    df['open_close'] = df['open'] - df['close']
    df['vol_change'] = df['volume'].pct_change()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['std5'] = df['close'].rolling(5).std()
    df['std10'] = df['close'].rolling(10).std()
    df.dropna(inplace=True)
    
    features = ['return', 'high_low', 'open_close', 'vol_change', 'ma5', 'ma10', 'std5', 'std10']
    return df[features]

def predict_next_day(ticker):
    df = fetch_data(ticker)
    features = create_features(df)
    latest = features.iloc[-1:]
    scaled = scaler.transform(latest)
    prediction = model.predict(scaled)[0]
    trend = "UP" if prediction == 1 else "DOWN"
    print(f"Predicted trend for {ticker} tomorrow: {trend}")

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, TATASTEEL.NS): ")
    predict_next_day(ticker)
