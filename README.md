# Speech Emotion Recognition

A Flask web application that predicts emotion from an uploaded audio recording. It extracts 40 MFCC features with librosa and classifies them with a trained TensorFlow model and label encoder.

## Stack

- Python and Flask
- librosa and NumPy
- TensorFlow / Keras

## Setup

1. Use a Python version supported by the TensorFlow release installed in your environment and create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the application with `python app.py` and open the local address printed by Flask.
4. Upload an audio recording through the interface to receive a prediction.

The trained model and label encoder are stored in `Models/`. Audio is processed by the local application; no external API credentials are needed.
