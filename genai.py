import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables and configure Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load class names and disease information
class_indices = json.load(open('class_indices.json'))
disease_info = json.load(open('plant_disease_info.json'))

# Load the pre-trained model with error handling
try:
    model = tf.keras.models.load_model('trained_model/plant_disease_prediction_model.h5', compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Prevent further usage if loading fails

# Function to load and preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the disease class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)]

# Function to get disease information
def get_disease_info(predicted_class):
    return disease_info.get(predicted_class, {})

# Function to interact with Gemini AI
def ask_gemini(question):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 1000,
                "response_mime_type": "text/plain",
            }
        )
        
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(question)
        
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini AI: {e}"

# Streamlit App
if model is not None:
    st.title('AI for Farmers: Plant Disease Classifier')

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
                st.success(f'Prediction: {prediction}')
                disease_info = get_disease_info(prediction)

                st.write(f"**Plant Name:** {disease_info.get('plant_name')}")
                st.write(f"**Symptoms:** {disease_info.get('symptoms')}")
                st.write(f"**Causes:** {disease_info.get('causes')}")
                st.write(f"**Preventive Measures:** {disease_info.get('preventive_measures')}")
                st.write(f"**Treatment:** {disease_info.get('treatment')}")
    
    # Automatically query Gemini AI for more details about the disease
    ai_initial_response = ask_gemini(f"Tell me more about {prediction} in detail")
    st.subheader('AI Generated Detailed Information:')
    st.write(ai_initial_response)

    # Automatically ask AI about prevention techniques
    ai_prevention_response = ask_gemini(f"What are the detailed prevention techniques for {prediction}?")
    st.subheader('AI Generated Prevention Techniques:')
    st.write(ai_prevention_response)

    # Separate section for further Q&A with Gemini AI
    st.subheader('Ask AI more questions about the disease')
    user_question = st.text_input('Enter your question:')
    if st.button('Ask AI'):
        if user_question:
            ai_answer = ask_gemini(user_question)
            st.write(f"**AI Answer (Gemini):** {ai_answer}")
        else:
            st.write("Please enter a question.")
