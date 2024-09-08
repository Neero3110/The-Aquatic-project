import os
import tensorflow as tf
import requests

# Define a function to download your model from cloud storage
def download_model():
    if not os.path.exists('aquaticGuide.h5'):
        url = "your_model_download_link"
        response = requests.get("https://drive.google.com/file/d/1u2AW5jRlmCjClcJ7Xkdnzw0gAjbuvUfp/view?usp=sharing")
        with open("aquaticGuide.h5", 'wb') as f:
            f.write(response.content)

# Download the model
download_model()

# Load the model
model = tf.keras.models.load_model('aquaticGuide.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and saved as model.tflite")
