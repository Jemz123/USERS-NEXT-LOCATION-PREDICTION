import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Example dataset: List of (latitude, longitude) tuples
data = [
    (12.9716, 77.5946),  # Bangalore
    (13.0827, 80.2707),  # Chennai
    (28.7041, 77.1025),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    (22.5726, 88.3639),  # Kolkata
    # Add more locations here
]

# Convert the list into a DataFrame
df = pd.DataFrame(data, columns=['latitude', 'longitude'])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare the dataset for LSTM
# Define the function to create sequences of input-output pairs
def create_sequences(data, seq_length=1):
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
        
    return np.array(X), np.array(y)

# Create sequences of location data
seq_length = 2  # For example, using 2 previous locations to predict the next
X, y = create_sequences(scaled_data, seq_length)

# Reshape data to be compatible with LSTM input
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=2))  # Output layer with 2 units (latitude, longitude)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=1)

# Predict the next location
last_sequence = scaled_data[-seq_length:]  # Use the last 'seq_length' locations
last_sequence = last_sequence.reshape(1, seq_length, 2)  # Reshape for LSTM input

predicted_location = model.predict(last_sequence)
predicted_location = scaler.inverse_transform(predicted_location)  # Inverse scale to get original coordinates

print(f"Predicted next location: Latitude: {predicted_location[0][0]}, Longitude: {predicted_location[0][1]}")
