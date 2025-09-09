from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load your model
model = load_model("emotion_detector_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Read image and preprocess
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    pred = model.predict(img)
    label = emotion_labels[np.argmax(pred)]

    return render_template('index.html', prediction=label, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
