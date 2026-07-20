import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("stocks.csv")

# Normalize column names to lowercase
df.columns = df.columns.str.strip().str.lower()
print("✅ Columns found:", df.columns.tolist())

# If only 'adj close' exists, use it as 'close'
if 'close' not in df.columns and 'adj close' in df.columns:
    df['close'] = df['adj close']

# Feature Engineering
df['return'] = df['close'].pct_change()
df['high_low'] = df['high'] - df['low']
df['open_close'] = df['open'] - df['close']
df['vol_change'] = df['volume'].pct_change()
df['ma5'] = df['close'].rolling(5).mean()
df['ma10'] = df['close'].rolling(10).mean()
df['std5'] = df['close'].rolling(5).std()
df['std10'] = df['close'].rolling(10).std()
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop missing rows
df.dropna(inplace=True)

# Features and labels
features = ['return', 'high_low', 'open_close', 'vol_change', 'ma5', 'ma10', 'std5', 'std10']
X = df[features]
y = df['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training complete. Model and scaler saved.")
