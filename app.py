import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración inicial
st.set_page_config(page_title="Clasificador de Imágenes", layout="centered")

st.title("🧠 Clasificador de Imágenes con TensorFlow")
st.write("Subí una imagen para que el modelo te diga a qué clase pertenece.")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("Mejor_modelo_FINAL_20241126.keras")

modelo = cargar_modelo()

# Clases que predice el modelo (editá esto)
clases = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']
IMG_SIZE = (224, 224)

# Subida de imagen
imagen_subida = st.file_uploader("📷 Subí tu imagen aquí", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="Imagen subida", use_column_width=True)

    # Procesamiento
    imagen = imagen.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(imagen)
    img_array = tf.expand_dims(img_array, 0)

    # Predicción
    predicciones = modelo.predict(img_array)
    indice_predicho = np.argmax(predicciones)
    clase_predicha = clases[indice_predicho]
    confianza = np.max(predicciones)

    st.markdown(f"### 🎯 Predicción: **{clase_predicha}**")
    st.markdown(f"Confianza: `{confianza:.2f}`")