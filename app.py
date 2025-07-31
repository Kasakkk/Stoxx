import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Initialize session state for assistant messages and input tracking
if 'assistant_messages' not in st.session_state:
    st.session_state.assistant_messages = []
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = {}

# Function to add assistant messages
def add_assistant_message(message):
    if message not in st.session_state.assistant_messages:
        st.session_state.assistant_messages.append(message)

# Function to clear assistant messages
def clear_assistant_messages():
    st.session_state.assistant_messages = []

# Function to validate input
def validate_input(value, field, day):
    if value <= 0:
        add_assistant_message(f"⚠️ **{field} -{day}** is zero or negative. Please enter a positive value for accurate predictions.")
        return False
    if field == "Volume" and value < 1000:
        add_assistant_message(f"⚠️ **{field} -{day}** seems low ({value}). Typical stock volumes are in thousands or millions.")
        return False
    if field in ["Open", "High", "Low", "Close"] and (value < 0.01 or value > 10000):
        add_assistant_message(f"⚠️ **{field} -{day}** ({value}) is unusually extreme. Ensure the price is realistic.")
        return False
    return True

# Function to provide input-specific guidance
def provide_input_guidance(field, value, day):
    if field == "Open":
        add_assistant_message(f"**Open -{day}** ({value:.2f}): This is the starting price for the day. A higher open compared to the previous day's close suggests positive sentiment.")
    elif field == "High":
        add_assistant_message(f"**High -{day}** ({value:.2f}): The day's peak price. A large gap from the open indicates strong buying pressure.")
    elif field == "Low":
        add_assistant_message(f"**Low -{day}** ({value:.2f}): The day's lowest price. A narrow high-low range may suggest low volatility.")
    elif field == "Close":
        add_assistant_message(f"**Close -{day}** ({value:.2f}): The day's final price. It heavily influences the model's trend prediction.")
    elif field == "Volume":
        add_assistant_message(f"**Volume -{day}** ({value}): High volume can amplify price movements, affecting volatility features in the model.")

# Function to create features from input data
def create_features(data):
    df = pd.DataFrame([data[i:i+5] for i in range(0, len(data), 5)], 
                      columns=['open', 'high', 'low', 'close', 'volume'])
    df['return'] = df['close'].pct_change()
    df['high_low'] = df['high'] - df['low']
    df['open_close'] = df['open'] - df['close']
    df['vol_change'] = df['volume'].pct_change()
    df['ma5'] = df['close'].rolling(3).mean()  # Adjusted for 3 days
    df['ma10'] = df['close'].rolling(3).mean()  # Use 3 for consistency
    df['std5'] = df['close'].rolling(3).std()
    df['std10'] = df['close'].rolling(3).std()
    df.dropna(inplace=True)
    features = ['return', 'high_low', 'open_close', 'vol_change', 'ma5', 'ma10', 'std5', 'std10']
    return df[features].iloc[-1:].values

st.set_page_config(page_title="Stock Trend Predictor with Dynamic AI Assistant")
st.title("Next-Day Stock Prediction")

# AI Assistant Introduction
st.sidebar.title("Dynamic AI Assistant")
st.sidebar.markdown("I'm your AI Assistant, updating live as you enter stock data. I'll validate inputs, explain their impact, and guide you through predictions.")
add_assistant_message("Enter stock data for the past 3 trading days below. I'll provide feedback as you type.")

st.markdown("Enter **Open**, **High**, **Low**, **Close**, and **Volume** for the past 3 trading days (Day -3 is oldest):")

features = []
for i in [3, 2, 1]:
    st.subheader(f"Day -{i}")
    o = st.number_input(f"Open -{i}", value=0.0, format="%.2f", key=f"o{i}")
    h = st.number_input(f"High -{i}", value=0.0, format="%.2f", key=f"h{i}")
    l = st.number_input(f"Low -{i}", value=0.0, format="%.2f", key=f"l{i}")
    c = st.number_input(f"Close -{i}", value=0.0, format="%.2f", key=f"c{i}")
    v = st.number_input(f"Volume -{i}", value=0, step=10000, key=f"v{i}")
    
    # Check for input changes
    input_key = f"day_{i}"
    current_inputs = {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
    if input_key not in st.session_state.last_inputs or st.session_state.last_inputs[input_key] != current_inputs:
        clear_assistant_messages()  # Clear previous messages on change
        add_assistant_message(f"Updating inputs for Day -{i}...")
        for field, value in current_inputs.items():
            if validate_input(value, field.capitalize(), i):
                provide_input_guidance(field.capitalize(), value, i)
        st.session_state.last_inputs[input_key] = current_inputs
    
    features += [o, h, l, c, v]

# Display assistant messages in sidebar
for msg in st.session_state.assistant_messages:
    st.sidebar.markdown(msg)

if st.button("Predict"):
    try:
        # Load model and scaler
        clf = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Create and scale features
        x = create_features(features)
        if x.shape[0] == 0:
            raise ValueError("Not enough valid data to compute features.")
        x_scaled = scaler.transform(x)

        # Make prediction
        trend_prediction = clf.predict(x_scaled)[0]
        probability = clf.predict_proba(x_scaled)[0][1]
        last_close = features[-2]  # Close price of Day -1
        estimated_pct = (clf.predict_proba(x_scaled)[0][1] - 0.5) * 10  # Heuristic estimate

        st.subheader("Prediction Result")
        trend = "UP" if trend_prediction == 1 else "DOWN"
        st.write(f"**Predicted Direction**: {trend}")
        st.write(f"**Confidence Level**: {probability:.2%}")
        st.write(f"**Estimated Percentage Change**: {estimated_pct:.2f}%")

        # Assistant interpretation
        clear_assistant_messages()  # Clear input messages for prediction
        add_assistant_message(f"🎯 **Prediction**: The model predicts a {trend} trend for tomorrow with {probability:.2%} confidence.")
        add_assistant_message(f"This suggests a {probability:.2%} chance the stock price will {'increase' if trend == 'UP' else 'decrease'} from the last close (${last_close:.2f}).")
        add_assistant_message(f"The estimated change is {estimated_pct:.2f}%, based on model confidence.")
        
        # Action suggestion
        if trend == "UP" and probability > 0.7:
            add_assistant_message("A strong upward trend might indicate a buying opportunity, but please consult a financial advisor.")
        elif trend == "DOWN" and probability > 0.7:
            add_assistant_message("A strong downward trend suggests caution or a potential sell, but consult a financial advisor.")
        else:
            add_assistant_message("Moderate confidence suggests holding and monitoring. Always consult a financial advisor.")

        # Redisplay assistant messages after prediction
        for msg in st.session_state.assistant_messages:
            st.sidebar.markdown(msg)

    except Exception as e:
        st.subheader("Prediction Result")
        st.write("Could not generate prediction. Please ensure all fields are filled correctly.")
        clear_assistant_messages()
        add_assistant_message(f"❌ **Error**: {str(e)}. Ensure all inputs are valid and non-zero. Try re-entering the data.")