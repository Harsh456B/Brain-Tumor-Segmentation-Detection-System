import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from utilities import focal_tversky, tversky, tversky_loss, prediction

# Set Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Load the ResUNet Model
@st.cache_resource
def load_model():
    with open('ResUNet-MRI.json', 'r') as json_file:
        json_saved_model = json_file.read()
    model = tf.keras.models.model_from_json(json_saved_model, custom_objects={'Model': tf.keras.models.Model})
    model.load_weights('C:/Users/VICTUS/Desktop/Brain456/Brain_MRI/weights_seg.hdf5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    model.compile(optimizer=optimizer, loss=focal_tversky, metrics=[tversky])
    return model

model_seg = load_model()

# Title and description
st.title("🧠 Brain Tumor Detection from MRI")
st.markdown("""
This AI-powered tool uses a **ResUNet deep learning model** to segment brain tumors from MRI images.  
Upload an MRI image and the model will highlight the tumor region if detected.
""")

# Sidebar
st.sidebar.header("📁 Upload MRI Image")
uploaded_file = st.sidebar.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

# Main display
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.subheader("📷 Original MRI Image")
    st.image(image, use_column_width=True)

    with st.spinner("Processing..."):
        # Preprocess the image
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (256, 256))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # Predict the mask
        predicted_mask = model_seg.predict(img_input)[0].squeeze().round()

    # Display the results
    st.subheader("🧪 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(img_resized, use_column_width=True)
    with col2:
        st.markdown("**Predicted Tumor Mask**")
        st.image(predicted_mask, use_column_width=True, clamp=True)

    # Diagnosis result
    if predicted_mask.any():
        st.success("✅ **Tumor Detected**")
    else:
        st.info("❌ **No Tumor Detected**")
else:
    st.warning("Please upload a brain MRI image to proceed.")
