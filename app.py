import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image  # Import PIL for image processing

# Load the model
model = load_model(r'C:\Users\Jeffr\OneDrive\Documents\Image Classification Project\Image_Classification_Tutorial.keras')

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 
    'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
    'pomegranate', 'potato', 'radish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_width = 180
img_height = 180

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = Image.open(uploaded_file)
    image_load = image_load.resize((img_width, img_height))  # Resize the image
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_batch = tf.expand_dims(img_arr, 0)  # Expanding dimensions to match model input

    # Prediction
    predict = model.predict(img_batch)
    score = tf.nn.softmax(predict)

    # Display the uploaded image
    st.image(image_load, caption='Uploaded Image', use_column_width=True)

    # Convert score to numpy array
    score = score.numpy()

    # Sort the results by probability and display the top 5
    sorted_results = sorted(zip(data_cat, score[0]), key=lambda x: x[1], reverse=True)

    # Display the top 5 results
    st.write("Top 5 Prediction Results:")
    for cat, per in sorted_results[:5]:  # Take only the first 5 results
        st.write(f"Category: {cat}, Probability: {100 * per:.4f}")
