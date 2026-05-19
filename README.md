# 🛸 CIFAR-10 · Clasificador de Imágenes CNN
### Taller: CNN y Fine-Tuning — Punto 5: Clasificación de Imagen Externa

---

## 📁 Estructura del Proyecto

```
CIFAR10_Clasificador/
│
├── models/
│   └── best_cifar10_model.keras   ← ⚠️ DEBES COLOCAR AQUÍ TU MODELO
│
├── templates/
│   └── index.html                 ← Interfaz web (cosmos + español)
│
├── app.py                         ← Servidor Flask (API + web)
├── predict_local.py               ← Script para Spyder/consola
├── requirements.txt               ← Dependencias
└── README.md
```

---

## 🚀 Instalación y Ejecución

### 1. Colocar el modelo entrenado
Copia tu archivo `best_cifar10_model.keras` (generado en Google Colab) dentro de la carpeta `models/`.

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor web
```bash
python app.py
```
Luego abre tu navegador en: **http://localhost:5000**

---

## 🖥️ Uso con Spyder (sin servidor web)

Para clasificar directamente desde consola o Spyder:
```bash
python predict_local.py ruta/de/tu/imagen.jpg
```
Si no pasas la ruta como argumento, el script la solicitará interactivamente.

---

## 🏷️ Clases detectables

| # | Español     | English    |
|---|-------------|------------|
| 0 | Avión       | airplane   |
| 1 | Automóvil   | automobile |
| 2 | Pájaro      | bird       |
| 3 | Gato        | cat        |
| 4 | Ciervo      | deer       |
| 5 | Perro       | dog        |
| 6 | Rana        | frog       |
| 7 | Caballo     | horse      |
| 8 | Barco       | ship       |
| 9 | Camión      | truck      |

---

## 📊 Características del Modelo

- **Arquitectura**: ResNet-style CNN
- **Precisión alcanzada**: 93%
- **Épocas de entrenamiento**: 120
- **Tamaño de entrada**: 32×32×3
- **Dataset**: CIFAR-10 (50,000 imágenes train / 10,000 test)
- **Técnicas de Fine-Tuning**:
  - Data Augmentation (flip, rotación, zoom, traslación)
  - Cosine Decay Learning Rate Schedule
  - Dropout progresivo (0.1 → 0.5)
  - Bloques residuales dobles por sección
  - BatchNormalization en cada capa conv

---

*Desarrollado con TensorFlow 2.x · Flask · HTML5 Canvas*
