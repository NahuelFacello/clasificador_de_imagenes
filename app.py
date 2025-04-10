import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import img_to_array, load_img
from PIL import Image

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("Mejor_modelo_FINAL_20241126.keras")

modelo = cargar_modelo()

# Lista de nombres de clases (ajustá con las tuyas reales)
nombres_clases = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
                  'perro', 'rana', 'caballo', 'barco', 'camión']

# Título
st.title("🧠 Clasificador de imágenes (32x32)")

# Subida de imagen
imagen_subida = st.file_uploader("📷 Subí una imagen", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    # Preprocesar imagen
    imagen = imagen.resize((32, 32))                         # Redimensionar a 32x32
    imagen_array = img_to_array(imagen) / 255.0              # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=0)      # Agregar dimensión batch

    # Predicción
    prediccion = modelo.predict(imagen_array)
    indice_prediccion = np.argmax(prediccion)
    clase_predicha = nombres_clases[indice_prediccion]

    # Mostrar resultado
    st.markdown(f"### 🎯 Clasificación: **{clase_predicha}** (Clase {indice_prediccion})")
