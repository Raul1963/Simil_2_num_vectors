import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Load the data
train_data = pd.read_csv('ROST-P/ROST-P-trainSet1.csv')  # Use your file path
test_data = pd.read_csv('ROST-P/ROST-P-testSet1.csv')   # Use your file path

# Split features (X) and labels (y)
X_train = train_data.iloc[:, :-1].values  # All columns except the last one (features)
y_train = train_data['authorCode'].values  # Last column (labels)

X_test = test_data.iloc[:, :-1].values  # Same for test data
y_test = test_data['authorCode'].values

# Binary classification: If 'authorCode' == 5, label as 1; otherwise, 0
y_train = np.where(y_train == 5, 1, 0)
y_test = np.where(y_test == 5, 1, 0)

# Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use long for classification
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))  # Number of classes in your data
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = SimpleANN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model( X_train_tensor)
    loss = criterion(outputs,y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_test = model(X_test_tensor)

# Get predicted class labels (highest score in each prediction)
_, predicted_train = torch.max(y_pred_train, 1)
_, predicted_test = torch.max(y_pred_test, 1)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, predicted_train.numpy())
test_accuracy = accuracy_score(y_test, predicted_test.numpy())

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

all_labels = np.concatenate((y_test, predicted_test.numpy()))
unique_labels = np.unique(all_labels)
cm = confusion_matrix(y_test, predicted_test.numpy(), labels=unique_labels)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
