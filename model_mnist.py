from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 input images into a 1D array
model.add(Dense(128, activation='relu'))  # Fully connected layer with 128 units and ReLU activation
model.add(Dense(10, activation='softmax'))  # Output layer with 10 units and softmax activation for 10 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Load the image to predict
img = image.load_img(Path(__file__).parent/"digit-8.png", target_size=(28, 28), color_mode="grayscale")

# Convert the image to an array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x.astype('float32') / 255.0

# Make the prediction
predicted_probabilities = model.predict(x)
predicted_label = np.argmax(predicted_probabilities, axis=-1)

print(predicted_label)

# Display the image and prediction
# plt.imshow(img, cmap='gray')
# plt.title(f"Predicted Label: {predicted_label[0]}")
# plt.show()
