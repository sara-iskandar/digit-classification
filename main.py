# Import libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# Convert labels to integers
y = y.astype(np.uint8)

# Split the dataset into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values to the range [0, 1]
print("Normalizing data...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train a Logistic Regression model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
print("Saving model...")
joblib.dump(model, "mnist_logistic_regression.pkl")

# Visualize some predictions
def plot_digit(image, label, prediction):
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"Label: {label}, Prediction: {prediction}")
    plt.axis("off")
    plt.show()

print("Visualizing predictions...")
for i in range(5):
    plot_digit(X_test.iloc[i], y_test.iloc[i], y_pred[i])