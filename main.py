import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))
disease_info_path = f"{working_dir}/plant_disease_info.json"
with open(disease_info_path, 'r') as f:
    disease_info = json.load(f)



def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)    
    img = img.resize(target_size)    
    img_array = np.array(img)    
    img_array = np.expand_dims(img_array, axis=0)    
    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def get_disease_info(predicted_class_name):
    return disease_info.get(predicted_class_name, {})

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

            disease_details = get_disease_info(prediction)
            if disease_details:
                st.write(f"Plant Affected: {disease_details['plant_name']}")
                st.write(f"Symptoms: {disease_details['symptoms']}")
                st.write(f"Causes: {disease_details['causes']}")
                st.write(f"Preventive Measures: {disease_details['preventive_measures']}")
                st.write(f"Treatment: {disease_details['treatment']}")
            else:
                st.error("No detailed information available for this disease.")