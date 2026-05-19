"""
predict_local.py — Punto 5: Clasificación de Imagen Externa
Ejecutar directamente en Spyder o consola.
Uso: python predict_local.py ruta/a/tu/imagen.jpg
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = os.path.join("models", "best_cifar10_model.keras")

CLASES = {
    0: ("Avión",       "airplane",   "✈️ "),
    1: ("Automóvil",   "automobile", "🚗"),
    2: ("Pájaro",      "bird",       "🐦"),
    3: ("Gato",        "cat",        "🐱"),
    4: ("Ciervo",      "deer",       "🦌"),
    5: ("Perro",       "dog",        "🐶"),
    6: ("Rana",        "frog",       "🐸"),
    7: ("Caballo",     "horse",      "🐴"),
    8: ("Barco",       "ship",       "🚢"),
    9: ("Camión",      "truck",      "🚚"),
}

# =====================================================
#  MODEL LOADING
# =====================================================
print("=" * 50)
print("  CIFAR-10 · Clasificador de Imagen")
print("=" * 50)

print(f"\nCargando modelo desde: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado correctamente.\n")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que 'best_cifar10_model.keras' esté en la carpeta /models/")
    sys.exit(1)

# =====================================================
# OBTENER RUTA DE IMAGEN
# =====================================================
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("Ruta de la imagen a clasificar: ").strip().strip('"')

if not os.path.isfile(img_path):
    print(f"No se encontró el archivo: {img_path}")
    sys.exit(1)

# =====================================================
# PREPROCESAR IMAGEN
# =====================================================
print(f"Procesando imagen...: {img_path}")
img = Image.open(img_path).convert("RGB")
print(f"Tamaño original : {img.size[0]}×{img.size[1]} px")

img_32 = img.resize((32, 32), Image.LANCZOS)
print(f"Redimensionada  : 32×32 px")

img_array = np.array(img_32, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

# =====================================================
# PREDICCION
# =====================================================
print("\nEjecutando red neuronal...")
predictions = model.predict(img_array, verbose=0)[0]

top3_indices = np.argsort(predictions)[::-1][:3]
best_idx     = top3_indices[0]
best_pct     = predictions[best_idx] * 100

# =====================================================
# RESULTADOS
# =====================================================
print("\n" + "=" * 50)
print(f"RESULTADO PRINCIPAL")
print("=" * 50)
emoji, nombre_es, nombre_en = CLASES[best_idx][2], CLASES[best_idx][0], CLASES[best_idx][1]
print(f"\n  {emoji}  Clase predicha : {nombre_es} ({nombre_en})")
print(f"      Confianza      : {best_pct:.2f}%")

print("\n  TOP 3 predicciones:")
print("  " + "─" * 36)
for rank, idx in enumerate(top3_indices):
    pct  = predictions[idx] * 100
    mark = "◀ ELEGIDA" if rank == 0 else ""
    print(f"  {rank+1}. {CLASES[idx][2]} {CLASES[idx][0]:<12} {pct:>6.2f}%  {mark}")

print("\n" + "=" * 50)
print("  Clasificación completada exitosamente.")
print("=" * 50 + "\n")
