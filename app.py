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
    st.write(f"Checkpoint descargado: {checkpoint_path}")
    return checkpoint_path

# Configurar el modelo SAM
@st.cache_resource
def load_sam_model(checkpoint_path):
    st.write("Cargando el modelo SAM...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Usando dispositivo: {device}")
        sam_model = sam_model_registry["vit_b"](checkpoint_path).to(device)
        predictor = SamPredictor(sam_model)
        st.write("Modelo SAM cargado correctamente.")
        return predictor
    except Exception as e:
        st.write(f"Error al cargar el modelo SAM: {e}")
        st.stop()

# Procesar una imagen redimensionada
def preprocess_image(image, max_size=512):
    """Redimensiona la imagen para minimizar la carga en el modelo SAM."""
    st.write("Redimensionando la imagen...")
    try:
        width, height = image.size
        scaling_factor = max_size / max(width, height)
        st.write(f"Factor de escalado: {scaling_factor}")
        if scaling_factor < 1:
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        st.write(f"Tamaño final de la imagen: {image.size}")
        return np.array(image)
    except Exception as e:
        st.write(f"Error durante la redimensión de la imagen: {e}")
        st.stop()

# Segmentar imagen
def segment_image(image, predictor):
    st.write("Segmentando la imagen...")
    try:
        image_np = preprocess_image(image)
        st.write(f"Tamaño de la imagen numpy: {image_np.shape}")
        predictor.set_image(image_np)

        # Coordenadas de una caja de ejemplo
        input_box = np.array([[50, 50, image_np.shape[1] - 50, image_np.shape[0] - 50]])
        st.write(f"Input box para segmentación: {input_box}")
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
        st.write("Segmentación completada.")
        
        # Forzar liberación de memoria de GPU si es necesario
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return masks[0]
    except Exception as e:
        st.write(f"Error durante la segmentación: {e}")
        st.stop()

# Configurar Streamlit
st.title("Webapp de Segmentación con SAM")
st.write("Este es un demo de segmentación usando SAM y Streamlit.")

# Descargar el modelo
st.write("Descargando checkpoint...")
checkpoint_path = download_checkpoint()

# Cargar el modelo
st.write("Cargando el modelo SAM...")
predictor = load_sam_model(checkpoint_path)

# Imagen por defecto
default_image_path = "default_image.jpg"  # Asegúrate de tener una imagen de prueba
if not os.path.exists(default_image_path):
    st.warning("Coloca una imagen llamada `default_image.jpg` en el directorio raíz.")
    st.stop()
else:
    st.write("Cargando imagen por defecto...")
    try:
        image = Image.open(default_image_path)
        st.image(image, caption="Imagen original")

        # Segmentar la imagen
        st.write("Iniciando segmentación...")
        mask = segment_image(image, predictor)

        # Mostrar la segmentación
        st.write("Máscara segmentada:")
        st.image(mask * 255, caption="Máscara generada", clamp=True)
    except Exception as e:
        st.write(f"Error al procesar la imagen: {e}")
        st.stop()

# Liberar memoria utilizada por PyTorch
if torch.cuda.is_available():
    torch.cuda.empty_cache()