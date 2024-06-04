from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)
model = None

class Adaline:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, X):
        return X.dot(self.weights) + self.bias

    def backward(self, X, y, y_pred, learning_rate):
        m = X.shape[0]
        dw = (2 / m) * X.T.dot(y_pred - y)
        db = (2 / m) * np.sum(y_pred - y, axis=0)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

def create_and_save_model():
    global model
    # Dummy data - replace with actual data loading and preprocessing
    data = np.random.rand(1000, 8)  # 1000 samples of size 8
    labels = np.random.randint(0, 36, 1000)  # 1000 labels for 36 classes (0-9, a-z)

    # Convert labels to one-hot encoding
    labels = np.eye(36)[labels]

    # Create the model
    model = Adaline(8, 36)

    # Train the model
    epochs = 1000
    learning_rate = 0.01
    for epoch in range(epochs):
        y_pred = model.forward(data)
        model.backward(data, labels, y_pred, learning_rate)

    # Save the model
    model_path = os.path.join(os.getcwd(), 'character_recognition_model.npz')
    np.savez(model_path, weights=model.weights, bias=model.bias)
    print(f"Model saved at {model_path}")

def load_model():
    global model
    model_path = os.path.join(os.getcwd(), 'character_recognition_model.npz')
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist. Creating a new one.")
        create_and_save_model()
    else:
        model_data = np.load(model_path)
        model = Adaline(8, 36)
        model.weights = model_data['weights']
        model.bias = model_data['bias']
        print("Model loaded successfully")

@app.route('/train', methods=['POST'])
def train():
    try:
        velocities = request.form.getlist('velocities')
        character = request.form['character']
        velocities = [float(v) for v in velocities]
        velocities = np.array(velocities)  # Keep velocities as a 1D array
        # Implement your training logic here
        # For demonstration, we just print the velocities and character
        print("Training with velocities:", velocities, "Character:", character)
        return jsonify({'message': 'Training complete'})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['POST'])
def test():
    try:
        if model is None:
            load_model()

        velocities = request.form.getlist('velocities')
        velocities = [float(v) for v in velocities]
        velocities = np.array(velocities)
        print(f"Velocities received: {velocities}")

        # Make prediction
        prediction = model.forward(velocities)
        print(f"Prediction raw output: {prediction}")
        predicted_class_index = np.argmax(prediction)
        print(f"Predicted class index: {predicted_class_index}")

        # Convert index to character
        possible_classes = list('0123456789abcdefghijklmnopqrstuvwxyz')
        predicted_class = possible_classes[predicted_class_index]
        print(f"Predicted class: {predicted_class}")

        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        print(f"Error during testing: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='127.0.0.1', port=5000)