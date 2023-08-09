import io

import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

from model_mnist import model

app = Flask(__name__)


def predict_digit(image_data):
    # Load the image to predict
    img = image.load_img(io.BytesIO(image_data), target_size=(28, 28), color_mode="grayscale")

    # Convert the image to an array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255.0

    # Make the prediction
    predicted_probabilities = model.predict(x)
    predicted_label = np.argmax(predicted_probabilities, axis=-1)

    return predicted_label[0]


@app.route('/predict/digit', methods=['POST'])
def post_predict():
    # Get image data as a 2D array from the JSON payload of the POST request
    # image_data = request.json.get('image_data')
    image_data = request.get_data()

    # Call the predict_digit function to obtain the predicted digit from the image
    predicted_digit = predict_digit(image_data)

    # Return the predicted digit as a JSON response
    return jsonify({'predicted_digit': str(predicted_digit)})


if __name__ == '__main__':
    app.run(port=8080)
