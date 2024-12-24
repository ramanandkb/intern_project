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
    image_load = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels
    st.image(image_load, caption='Uploaded Image', width=200)

    # Resize image to match the model's expected input size
    image_load = image_load.resize((img_width, img_height))
    img_arr = np.array(image_load) / 255.0  # Normalize pixel values
    img_bat = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    # Confirm input shape
    # st.write(f"Input shape: {img_bat.shape}")

    # Predict using the model
    try:
        predict = model.predict(img_bat)
        # st.write(f"Raw Prediction: {predict}")

        score = tf.nn.softmax(predict[0])  # Apply softmax to get probabilities

        # Determine the predicted class
        predicted_index = np.argmax(score)
        predicted_score = score[predicted_index]
        st.write(f"predicted score :{predicted_score}")

        # Classification threshold
        threshold = 0.01
        if predicted_score < threshold:
            st.markdown(
                "<span style='color: red; font-size: 20px; font-weight: bold;'>Unknown Person! "
                "Please upload another image.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"The Person is: <span style='color: green; font-size: 40px; font-weight: bold;'>"
                f"{data_cat[predicted_index]}</span>",
                unsafe_allow_html=True,
            )
            # st.write(f'Confidence: {predicted_score * 100:.2f}%')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
