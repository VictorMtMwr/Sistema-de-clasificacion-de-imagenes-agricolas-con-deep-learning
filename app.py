#Micriservicio de predicción de hojas
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask import Flask, render_template


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

modelo = tf.keras.models.load_model("model/modelo_hojas.h5")


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128)) 
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

SHAPES = ['Eliptica', 'Imparipinnada', 'Lanceolada', 'Obovada', 'Ovada', 'Palmeada', 'Trifoliada']

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file in request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    print(f"Received file of size: {len(image_bytes)} bytes")
    try:
        input_tensor = preprocess_image(image_bytes)
        prediction = modelo.predict(input_tensor)
        idx = int(np.argmax(prediction[0]))
        shape = SHAPES[idx]
        probability = float(prediction[0][idx])
        print(f"Prediction: {prediction}")
        return jsonify({
            "type": shape,
            "probability": probability
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
