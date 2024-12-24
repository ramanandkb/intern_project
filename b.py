import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Set up Streamlit UI
st.header('Image Classification Model')
data_cat = ['Akshay Kumar', 'Amitabh Bachchan', 'Prabhas', 'Vijay']
img_height = 128  # Update to match your model's expected height
img_width = 128   # Update to match your model's expected width

# Load the model
model_path = r'D:/infosys/imagesmodel/mymodel4.keras'
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = Image.open(uploaded_file)
    st.image(image_load, caption='Uploaded Image', width=200)

    # Resize image to match the model's expected input size
    image_load = image_load.resize((img_width, img_height))
    img_arr = tf.keras.preprocessing.image.img_to_array(image_load) / 255.0  # Normalize
    img_bat = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    # Predict using the model
    try:
        predict = model.predict(img_bat)  # Raw model predictions
        st.write("Raw Predictions:", predict[0])  # Show raw logits

        # Option 1: Direct confidence threshold
        predicted_index = np.argmax(predict[0])
        predicted_confidence = predict[0][predicted_index]
        threshold = 0.5  # Adjust threshold for classification

        if predicted_confidence < threshold:
            st.markdown(
                "<span style='color: red; font-size: 20px; font-weight: bold;'>Unknown Person! "
                "Please upload another image.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"The Person is: <span style='color: green; font-size: 20px; font-weight: bold;'>"
                f"{data_cat[predicted_index]}</span>",
                unsafe_allow_html=True,
            )
            st.write(f'Confidence: {predicted_confidence * 100:.2f}%')

        # Option 2: Top-k predictions
        st.subheader("Top-k Predictions")
        k = 3
        top_k_indices = np.argsort(predict[0])[-k:][::-1]  # Get indices of top-k predictions
        top_k_confidences = predict[0][top_k_indices]

        for i, idx in enumerate(top_k_indices):
            st.write(f"{i + 1}: {data_cat[idx]} - Confidence: {top_k_confidences[i] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
