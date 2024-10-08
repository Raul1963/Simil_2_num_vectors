import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from scikeras.wrappers import KerasClassifier
from scipy.stats import uniform

# Function to create model, required for KerasClassifier
def create_rnn_model(units=50, activation='tanh', optimizer='adam', dropout_rate=0.0):
    model = Sequential()
    model.add(SimpleRNN(units=units, activation=activation, input_shape=(1, X_train.shape[1])))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the data (train and test)
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
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Create a KerasClassifier for RandomizedSearchCV
model = KerasClassifier(build_fn=create_rnn_model, verbose=0)

# Define the hyperparameter space
param_dist = {
    'model__units': [30, 50, 70, 100],  # Number of units in the RNN layer
    'model__activation': ['relu', 'tanh'],  # Activation function
    'optimizer': ['adam', 'rmsprop'],  # Optimizer
    'batch_size': [16, 32, 64],  # Batch size
    'epochs': [10, 20, 30],  # Number of epochs
    'model__dropout_rate': uniform(0, 0.5),  # Dropout rate between 0 and 0.5
}

# Randomized search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                   n_iter=10, cv=3, verbose=1, random_state=42)

# Fit the model
random_search.fit(X_train_reshaped, y_train)

# Print the best parameters found
print("Best parameters found: ", random_search.best_params_)

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
y_pred = (best_model.predict(X_test_reshaped) > 0.5).astype(int)

# Accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
