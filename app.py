#from fastapi import FastAPI
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, request, jsonify
#from fastapi.responses import JSONResponse
import os
import tensorflow as tf
import requests
from PIL import Image
import numpy as np
import io

# Initialize the fastapi app
#app = FastAPI()
app = Flask(__name__)
asgi_app = WsgiToAsgi(app)

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

# Create an API route for image prediction
@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    
    file = request.files['file']

    # Run prediction
    result = predict(file)

    # Return the prediction result as JSON
    return jsonify({"prediction": result.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
