# AI for Farmers: Plant Disease Classifier

This project is a **Plant Disease Detection System** designed to assist farmers by identifying diseases in plants from uploaded images and providing detailed information and preventive measures. It integrates a pre-trained deep learning model with **Google Gemini AI** for enhanced interactivity and additional insights.

## Features

- **Image-based Disease Classification**: Uses a CNN model to predict plant diseases from uploaded images.
- **Disease Information Retrieval**: Provides details about the disease, including plant name, symptoms, causes, preventive measures, and treatments.
- **AI-Powered Q&A**: Leverages Google Gemini AI to answer user queries about plant diseases in detail.

## Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- Streamlit
- Google Generative AI Python SDK (`google-generativeai`)
- `.env` file containing `GOOGLE_API_KEY`

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name/plant-disease-classifier.git
   cd plant-disease-classifier
