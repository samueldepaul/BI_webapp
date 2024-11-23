import streamlit as st
import os
import gdown
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

# URL de Google Drive para descargar el checkpoint
checkpoint_url = "https://drive.google.com/uc?id=1bhA4qPxEcvo1jq8sajIY8yg0ZjWPMVYZ"
checkpoint_path = "sam_vit_b_01ec64.pth"

# Descargar el checkpoint si no está presente
@st.cache_resource
def download_checkpoint():
    if not os.path.exists(checkpoint_path):
        st.write("Descargando el modelo SAM...")
        gdown.download(checkpoint_url, checkpoint_path, quiet=False)
    return checkpoint_path

# Configurar el modelo SAM
@st.cache_resource
def load_sam_model(checkpoint_path):
    st.write("Cargando el modelo SAM...")
    sam_model = sam_model_registry["vit_b"](checkpoint_path).to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam_model)
    return predictor

# Procesar una imagen y asegurar que está en formato RGB
def prepare_image(image):
    """
    Asegura que la imagen tenga 3 canales (RGB) para ser compatible con SAM.
    """
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # Escala de grises
        st.warning("La imagen está en escala de grises. Convirtiendo a RGB...")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # Tiene canal alfa (RGBA)
        st.warning("La imagen tiene un canal alfa. Eliminándolo...")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    return image_np

# Segmentar la imagen
def segment_image(image, predictor):
    st.write("Segmentando la imagen...")
    image_np = prepare_image(image)  # Asegurar que la imagen tenga 3 canales
    predictor.set_image(image_np)

    # Coordenadas de una caja de ejemplo
    input_box = np.array([[50, 50, image_np.shape[1] - 50, image_np.shape[0] - 50]])
    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
    return masks[0]

# Configurar Streamlit
st.title("Webapp de Segmentación con SAM")
st.write("Este es un demo de segmentación usando SAM y Streamlit.")

# Descargar el modelo
checkpoint_path = download_checkpoint()

# Cargar el modelo
predictor = load_sam_model(checkpoint_path)

# Imagen por defecto
default_image_path = "default_image.jpg"  # Asegúrate de tener una imagen de prueba
if not os.path.exists(default_image_path):
    st.warning("Coloca una imagen llamada `default_image.jpg` en el directorio raíz.")
else:
    image = Image.open(default_image_path)
    st.image(image, caption="Imagen original")

    # Segmentar la imagen
    mask = segment_image(image, predictor)

    # Mostrar la segmentación
    st.write("Máscara segmentada:")
    st.image(mask * 255, caption="Máscara generada", clamp=True)