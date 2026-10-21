from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import tensorflow as tf
import pickle
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings

# Disable oneDNN optimizations (optional for numerical consistency)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Debugging information
# Load the trained model
try:
    model = tf.keras.models.load_model(Path(__file__).resolve().parent / 'Models' / 'emotion_recognition_model.h5')
    # Compile the model to suppress warnings about `model.compile_metrics`
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[])
    print("Model loaded and compiled successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

# Load the label encoder
try:
    with open(Path(__file__).resolve().parent / 'Models' / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading label encoder: {str(e)}")
    raise e

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    try:
        return render_template('index.html')  # Ensure index.html exists in the `templates` folder
    except Exception as e:
        return jsonify({'error': 'Error loading the homepage: ' + str(e)})

# Emotion prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Load audio file
        data, sample_rate = librosa.load(file, sr=22050)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

        # Make prediction
        predictions = model.predict(mfccs_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

        return jsonify({'emotion': predicted_label[0]})

    except Exception as e:
        return jsonify({'error': 'Error during prediction: ' + str(e)})

if __name__ == '__main__':
    # Running the Flask application
    app.run(debug=False)
