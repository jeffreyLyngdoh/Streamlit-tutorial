
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import PIL
from PIL import Image  



st.title('This is just a test made a change')

# Try to load the model and handle any errors
model = None
try:
    model = load_model('./Image_Classification_Tutorial.keras')
    st.write("Model loaded successfully")
except Exception as e:
    st.write(f"Error loading model: {e}")



uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.write("File uploaded")
