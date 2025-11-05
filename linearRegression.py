import numpy as np
import matplotlib.pyplot as plt
# Step 1: Create simple dataset
# Hours studied vs. exam score
X = np.array([1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 2.5, 3.5, 4.5, 5.5, 7.5], dtype=float)

y = np.array([2.2, 3.8, 5.1, 6.1, 7.2,
 7.8, 8.5, 9.0, 9.5, 10.5,
 4.5, 5.5, 6.6, 6.9, 8.9], dtype=float)


w = 0.0  # weight (slope)
b = 0.0  # bias (intercept)
learning_rate = 0.01
epochs = 10000
# Step 2: Define the linear regression model

def model(X, w, b):
    return w * X + b
# Step 3: Define the loss function (Mean Squared Error)

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 4: Train the model using Gradient Descent

for _ in range(epochs):
    y_pred = model(X, w, b)
    loss = compute_loss(y, y_pred)


    # Compute gradients
    dw = (2 / len(X)) * np.sum(X * (y - y_pred))
    db = (2 / len(X)) * np.sum(y - y_pred)

    # Update parameters
    w += learning_rate * dw
    b += learning_rate * db

print("Learned w:", w)
print("Learned b:", b)
print("Prediction for x=6:", model(6, w, b))
print("Loss", loss)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model(X, w, b), color='red', label='Fitted line')
plt.show()

