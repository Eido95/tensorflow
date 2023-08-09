import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = None


# Prediction step
def make_prediction(model, new_data):
    predictions = model.predict(new_data)
    return predictions


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert the data to a NumPy array
        new_data = np.array(data['features'])

        # Make predictions using the trained model
        predictions = make_prediction(model, new_data)
        
        # Convert predictions to a list and return as JSON response
        response = {'predictions': predictions.tolist()}

        return jsonify(response), 200
    except Exception as exception:
        return jsonify({'error': str(exception)}), 500


if __name__ == '__main__':
    app.run(port=8080)
