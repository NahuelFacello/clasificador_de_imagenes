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

iconos = {
    'avión': '✈️',
    'automóvil': '🚗',
    'pájaro': '🐦',
    'gato': '🐱',
    'ciervo': '🦌',
    'perro': '🐶',
    'rana': '🐸',
    'caballo': '🐴',
    'barco': '🚢',
    'camión': '🚚'
}

# Título
st.title("🧠 Clasificador de imágenes")

st.write("")
# Explicación
st.markdown(
    """
    Esta aplicación utiliza un modelo de aprendizaje automático entrenado con imágenes de 10 categorías conocidas para predecir a qué clase pertenece una imagen cargada por el usuario.

    📌 **Importante**: Este clasificador solo puede identificar imágenes que correspondan a las siguientes categorías:

    - ✈️ avión
    - 🚗 automóvil
    - 🐦 pájaro
    - 🐱 gato
    - 🦌 ciervo
    - 🐶 perro
    - 🐸 rana
    - 🐴 caballo
    - 🚢 barco
    - 🚚 camión

    ✅ El modelo fue entrenado con un conjunto de datos balanceado y alcanza una **precisión del 95%** sobre los datos de prueba.

    ⚠️ Si subís imágenes que no pertenezcan a estas clases, el modelo puede predecir incorrectamente la categoría más parecida.
    """,
    unsafe_allow_html=True
)
st.write("")
# Subida de imagen
if imagen_subida is not None:
    imagen = Image.open(imagen_subida).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    # Preprocesamiento exacto que espera el modelo
    imagen = imagen.resize((32, 32))  # Tu modelo fue entrenado con 32x32
    imagen_array = img_to_array(imagen) / 255.0  # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=0)  # (1, 32, 32, 3)
    imagen_array = imagen_array.reshape(1, -1)  # (1, 3072)

    # Predicción
    prediccion = modelo.predict(imagen_array)
    indice_prediccion = np.argmax(prediccion)
    clase_predicha = nombres_clases[indice_prediccion]
    emoji = iconos.get(clase_predicha, '')

    st.markdown(f"### 🎯 Clasificación: **{emoji} {clase_predicha}** (Clase {indice_prediccion})")
    

st.write("")
st.write("")

st.markdown("---")
st.markdown("#### 📬 Contacto")

st.markdown(
    """
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
        <a href="https://github.com/NahuelFacello" target="_blank">
            <button style="padding: 10px 20px; font-size: 16px; border-radius: 8px; border: none; background-color: #24292e; color: white; cursor: pointer;">
                💻 GitHub
            </button>
        </a>
        <a href="https://www.linkedin.com/in/nahuel-facello/" target="_blank">
            <button style="padding: 10px 20px; font-size: 16px; border-radius: 8px; border: none; background-color: #0072b1; color: white; cursor: pointer;">
                🔗 LinkedIn
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
