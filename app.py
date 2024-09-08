import os
import tensorflow as tf
import requests
from PIL import Image
import numpy as np
import io

# Define a function to download your model from cloud storage
def download_model():
    if not os.path.exists('aquaticGuide.h5'):
        url = "https://drive.google.com/uc?export=download&id=1u2AW5jRlmCjClcJ7Xkdnzw0gAjbuvUfp"  # Use direct download URL
        response = requests.get(url)
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
    
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define a function for predictions
def predict(image_path):
    # Load and preprocess image
    image = Image.open(image_path).resize((224, 224))  # Adjust size as needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# Example usage
image_path = 'path_to_your_image.jpg'  # Replace with your image file path
predictions = predict(image_path)
print(predictions)
