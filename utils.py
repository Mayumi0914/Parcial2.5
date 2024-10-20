from diffusers import DiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import streamlit as st
from PIL import Image

def load_models():
    if "model_gen" not in st.session_state:
        # Carga el modelo de generación de imágenes desde diffusers
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        model_gen = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        model_gen.to("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.model_gen = model_gen

    if "model_class" not in st.session_state:
        # Carga el modelo de clasificación de imágenes desde transformers
        model_class = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        st.session_state.model_class = model_class
        st.session_state.processor = processor

def generate_image(prompt):
    if "model_gen" not in st.session_state:
        st.error("El modelo de generación no está cargado.")
        return None

    if not prompt:
        st.error("El prompt no puede estar vacío.")
        return None

    # Genera la imagen con el modelo de Diffusion
    with torch.no_grad():
        image = st.session_state.model_gen(prompt, num_inference_steps=25).images[0]
    return image

def classify_image(image):
    if "model_class" not in st.session_state or "processor" not in st.session_state:
        st.error("El modelo de clasificación no está cargado.")
        return None

    # Procesa y clasifica la imagen usando el modelo ResNet
    processor = st.session_state.processor
    model_class = st.session_state.model_class
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        logits = model_class(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model_class.config.id2label[predicted_label]
