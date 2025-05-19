import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import pandas as pd
import os

# --- Load the trained model ---
model = load_model('plant_disease_model.h5')

# --- Define class names ---
CLASS_NAMES = ('Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterialblight')

# --- Load CSV files with encoding fallback ---
def load_csv_with_fallback(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')
    return df

disease_df = load_csv_with_fallback('disease_info.csv')
supplement_df = load_csv_with_fallback('supplement_info.csv')

# --- Create lookup dictionaries ---
disease_info = disease_df.set_index('disease_class')['protection_tips'].to_dict()
supplement_info = supplement_df.set_index('disease_class')[['fertilizer', 'fertilizer_image']].to_dict(orient='index')

# --- Streamlit App ---
st.title("🌿 Plant Disease Detection System")
st.markdown("Upload an image of a plant leaf (JPG, PNG, or JFIF formats supported).")

# Allow jpg, png, jfif file types
plant_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jfif"])
submit = st.button('Predict Disease')

if submit:
    if plant_image is not None:
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            st.error("Error loading image. Please upload a valid image file (jpg, png, jfif).")
        else:
            # Show original image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")
            st.write(f"Original image shape: {opencv_image.shape}")

            # Preprocess image for model
            resized_image = cv2.resize(opencv_image, (256, 256))
            normalized_image = resized_image / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)

            # Predict disease class
            predictions = model.predict(input_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]

            # Display prediction result
            parts = predicted_class.split('___')
            if len(parts) == 2:
                st.title(f"🩺 This is a {parts[0]} leaf with {parts[1]}")
            else:
                st.title(f"Prediction: {predicted_class}")

     

            # Show protection tips
            protection_tips = disease_info.get(predicted_class)
            if protection_tips:
                st.subheader("🛡 Protection Tips:")
                st.write(protection_tips)
            else:
                st.warning("No protection tips found for this disease.")

            # Show fertilizer info and image
            supplement = supplement_info.get(predicted_class)
            if supplement:
                st.subheader("💊 Recommended Fertilizer:")
                st.write(supplement['fertilizer'])

                # Load local fertilizer image
                fertilizer_image_path = supplement['fertilizer_image']
                if os.path.exists(fertilizer_image_path):
                    st.image(fertilizer_image_path, caption="Fertilizer Image", width=300)
                else:
                    st.warning(f"Fertilizer image not found at path: {fertilizer_image_path}")
            else:
                st.warning("No supplement info found for this disease.")
    else:
        st.warning("Please upload a valid image file before clicking Predict.")
