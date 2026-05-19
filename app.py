"""
app.py - Servidor Flask para clasificación de imágenes CIFAR-10
Taller CNN y Fine-Tuning - Punto 5: Clasificación de Imagen Externa
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# =====================================================
# CLASS CIFAR-10 
# =====================================================
CLASES = {
    0: {"es": "Avión",        "en": "airplane",   "emoji": "✈️"},
    1: {"es": "Automóvil",    "en": "automobile",  "emoji": "🚗"},
    2: {"es": "Pájaro",       "en": "bird",        "emoji": "🐦"},
    3: {"es": "Gato",         "en": "cat",         "emoji": "🐱"},
    4: {"es": "Ciervo",       "en": "deer",        "emoji": "🦌"},
    5: {"es": "Perro",        "en": "dog",         "emoji": "🐶"},
    6: {"es": "Rana",         "en": "frog",        "emoji": "🐸"},
    7: {"es": "Caballo",      "en": "horse",       "emoji": "🐴"},
    8: {"es": "Barco",        "en": "ship",        "emoji": "🚢"},
    9: {"es": "Camión",       "en": "truck",       "emoji": "🚚"},
}

# =====================================================
# MODEL LOADING
# =====================================================
MODEL_PATH = os.path.join("models", "best_cifar10_model.keras")
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f" El Modelo se ha cargado correctamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"No se pudo cargar el modelo: {e}")
        print("Asegúrate de que 'best_cifar10_model.keras' esté en la carpeta /models/")

load_model()

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(image_bytes):

#Esta sección se encarga de cargar la imagen, redimensionarla a 32x32 px y normalizar los valores de los pixeles.

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
    return img_array

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Modelo no cargado. Verifica que 'best_cifar10_model.keras' esté en /models/"
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Archivo vacío."}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)

        predictions = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx]) * 100

        # Top 3 predicciones
        top3_indices = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                "clase_es": CLASES[i]["es"],
                "clase_en": CLASES[i]["en"],
                "emoji": CLASES[i]["emoji"],
                "confianza": round(float(predictions[i]) * 100, 2)
            }
            for i in top3_indices
        ]

        # Imagen preview en base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = file.content_type or "image/jpeg"

        return jsonify({
            "clase_es":   CLASES[top_idx]["es"],
            "clase_en":   CLASES[top_idx]["en"],
            "emoji":      CLASES[top_idx]["emoji"],
            "confianza":  round(confidence, 2),
            "top3":       top3,
            "imagen_b64": f"data:{ext};base64,{img_b64}"
        })

    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@app.route("/status")
def status():
    return jsonify({
        "modelo_cargado": model is not None,
        "modelo_path": MODEL_PATH,
        "clases": [CLASES[i]["es"] for i in range(10)]
    })


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
