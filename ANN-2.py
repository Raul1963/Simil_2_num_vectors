import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
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
# y_train = np.where(y_train == 5, 1, 0)
# y_test = np.where(y_test == 5, 1, 0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to categorical (not necessary for binary classification but useful for multi-class problems)
# y_train = to_categorical(y_train, 2)
# y_test = to_categorical(y_test, 2)

# Build the ANN model
model = Sequential()

# Add input layer and first hidden layer
model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))  # 16 neurons, relu activation

# Add second hidden layer
model.add(Dense(units=8, activation='relu'))  # 8 neurons, relu activation

# Add output layer for binary classification
model.add(Dense(units=1, activation='sigmoid'))  # 1 neuron, sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model on test data
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred,labels=np.unique([y_test,y_pred]))
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique([y_test,y_pred]))
disp.plot()
plt.show()
