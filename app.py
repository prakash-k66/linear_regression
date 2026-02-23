import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input (house size)
X = np.array([500, 800, 1000, 1200, 1500]).reshape(-1, 1)

# Output (house price in lakhs)
y = np.array([5, 8, 10, 12, 15])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[1100]])

print("Predicted price:", prediction[0])

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()