# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import traceback

app = Flask(__name__)
CORS(app)

# Asegura la carpeta de imágenes
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo
modelo = tf.keras.models.load_model("model/modelo_hojas.h5")

# Clases
SHAPES = ['Eliptica', 'Imparipinnada', 'Lanceolada', 'Obovada', 'Ovada', 'Palmeada', 'Trifoliada']

# Página principal
@app.route('/')
def home():
    return render_template('index.html', prediction=None, image=None)

# Preprocesamiento de imagen
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128)) 
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Ruta de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No se subió archivo", image=None)

    file = request.files["file"]
    image_bytes = file.read()

    # Guardar imagen para mostrarla en el HTML
    image_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    try:
        input_tensor = preprocess_image(image_bytes)
        prediction = modelo.predict(input_tensor)
        idx = int(np.argmax(prediction[0]))
        shape = SHAPES[idx]
        probability = float(prediction[0][idx])
        return render_template(
            "index.html",
            image=image_path.replace("\\", "/"),
            clase=shape,
            probabilidad=f"{probability*100:.2f}",
            prediction=None
        )
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", image=image_path.replace("\\", "/"), clase=None, probabilidad=None, prediction="Error en la predicción")

if __name__ == "__main__":
    app.run(debug=True)