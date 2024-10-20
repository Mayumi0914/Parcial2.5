import streamlit as st
from utils import load_models, generate_image, classify_image
from PIL import Image

st.set_page_config(layout="wide")

# Cargar los modelos
load_models()

st.title("Generación y Clasificación de Imágenes con HuggingFace")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
         st.header("Generador de Imágenes")
    # Input para el prompt de la generación de imagen
         prompt = st.text_input("Introduce una descripción para generar una imagen")
    
    # Botón para generar la imagen
         if st.button("Generar Imagen"):
             if prompt:
                 image = generate_image(prompt)
                 if image is not None:
                      st.image(image, caption="Imagen Generada")
                      st.session_state.generated_image = image
             else:
                  st.warning("Por favor, introduce una descripción.")
    
    # Botón para clasificar la imagen generada
         if "generated_image" in st.session_state:
             if st.button("Clasificar Imagen Generada"):
                 image = st.session_state.generated_image
                 prediction = classify_image(image)
                 if prediction:
                     st.write(f"Predicción: {prediction}")
         else:
              st.info("Primero genera una imagen para poder clasificarla.")



    # Columna 2: Clasificador de Imágenes
    with col2:
        st.header("Clasificador de Imágenes")
        uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Clasificación de imagen subida
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen subida")
            prediction = classify_image(image)
            if prediction:
                st.write(f"Predicción: {prediction}")
        elif "generated_image" in st.session_state:
            # Clasificación de la imagen generada
            if st.button("Clasificar Imagen Generada"):
                image = st.session_state.generated_image
                prediction = classify_image(image)
                if prediction:
                    st.write(f"Predicción: {prediction}")
        else:
            st.warning("Sube una imagen o genera una para clasificar.")
