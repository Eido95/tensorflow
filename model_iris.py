import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target.reshape(-1, 1)

# One-hot encode the labels for binary classification
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a simple feedforward neural network using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Two output neurons for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# ... (previous code remains the same)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Prediction step
def make_prediction(model, new_data):
    predictions = model.predict(new_data)
    return predictions

# Sample new data for prediction (replace this with your own data)
new_data = [[5.1, 3.5, 1.4, 0.2],  # Sample data point 1
            [6.3, 2.9, 5.6, 1.8],  # Sample data point 2
            # Add more data points here if needed
           ]

# Make predictions on new data
predictions = make_prediction(model, new_data)
print("Predictions on new data:")
print(predictions)



# {
#   "features": [
#     [5.1, 3.5, 1.4, 0.2],  // Sample data point 1
#     [6.3, 2.9, 5.6, 1.8],  // Sample data point 2
#     // Add more data points here if needed
#   ]
# }