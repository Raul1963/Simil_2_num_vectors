import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras import layers,ops
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv('ROST-P/Custom_train01.csv')  # Use your file path
test_data = pd.read_csv('ROST-P/Custom_test01.csv')   # Use your file path

# Split features (X) and labels (y)
X_train = train_data.iloc[:, :-1].values  # All columns except the last one (features)
y_train = train_data['authorCode'].values  # Last column (labels)

X_test = test_data.iloc[:, :-1].values  # Same for test data
y_test = test_data['authorCode'].values

# Binary classification: If 'authorCode' == 5, label as 1; otherwise, 0
y_train = np.where(y_train == 5, 1, 0)
y_test = np.where(y_test == 5, 1, 0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data to 3D for RNN (samples, timesteps, features)
# Here timesteps will be 1, as we don't have explicit time-series data but need to add that dimension for RNN
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the RNN model
model = Sequential()

# Add RNN layer
model.add(SimpleRNN(units=50, activation='tanh', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))

# Add output layer for binary classification
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate the model
y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not 5", "Is 5"])
disp.plot()
plt.show()
