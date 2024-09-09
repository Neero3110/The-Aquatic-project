import os
import tensorflow as tf
import requests
from PIL import Image
import numpy as np
import io

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="fish-detect.tflite")
interpreter.allocate_tensors()

# Define a function for predictions
def predict(image_path):
    # Load and preprocess image
    image = Image.open(image_path).resize((150, 150))  # Adjust size as needed
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
image_path = '1c06f5e8-2520-40d0-85bd-6f80d041dfb8-840mm.jpg'  # Replace with your image file path
predictions = predict(image_path)
print(predictions)
print('done')
