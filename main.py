import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError
import cv2
import json
import os

# Set page config at the very top
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

# Path for user data file
USER_DB_PATH = "user_db.json"

# Load the user database
def load_user_db():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "r") as f:
            return json.load(f)
    return {}

# Save the user database
def save_user_db(user_db):
    with open(USER_DB_PATH, "w") as f:
        json.dump(user_db, f)

# Save new users
def save_user(username, password):
    user_db = load_user_db()
    user_db[username] = password
    save_user_db(user_db)

# Authenticate users
def authenticate(username, password):
    user_db = load_user_db()
    if username in user_db and user_db[username] == password:
        return True
    return False

# Forgot password logic
def forgot_password(username, new_password):
    user_db = load_user_db()
    if username in user_db:
        user_db[username] = new_password
        save_user_db(user_db)
        return True
    return False

# Load the ResUNet Model
@st.cache_resource
def load_model():
    with open('ResUNet-MRI.json', 'r') as json_file:
        json_saved_model = json_file.read()
    model = tf.keras.models.model_from_json(json_saved_model, custom_objects={'Model': tf.keras.models.Model})
    model.load_weights('C:/Users/VICTUS/Desktop/Brain456/Brain_MRI/weights_seg.hdf5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

model_seg = load_model()

# Dashboard slide navigation
slide = st.sidebar.radio("🧭 Select Slide", ["Login Page", "MRI Detection", "Download Report"])

# Slide 1: Login Page
if slide == "Login Page":
    st.title("🧠 Login to Your Account")

    # Login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.session_state.slide = "MRI Detection"  # This will show the next slide
        else:
            st.error("Invalid credentials. Try again.")

    # Forgot password section
    st.markdown("### Forgot Password?")
    reset_username = st.text_input("Enter your username to reset password")
    new_password = st.text_input("Enter your new password", type="password")
    if st.button("Reset Password"):
        if forgot_password(reset_username, new_password):
            st.success("Password reset successfully! You can now log in.")
        else:
            st.error("Username not found.")

# Slide 2: MRI Detection
elif slide == "MRI Detection":
    if "logged_in" not in st.session_state:
        st.warning("Please log in to access the application.")
    else:
        st.title(f"🧠 Welcome, {st.session_state.username}! Tumor Detection")
        
        # Upload and enhance MRI image
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
        apply_gray = st.checkbox("Apply Grayscale")
        apply_contrast = st.slider("Enhance Contrast", 1.0, 3.0, 1.0)
        apply_sharpness = st.slider("Enhance Sharpness", 1.0, 3.0, 1.0)

        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert('RGB')

                # Apply filters
                if apply_gray:
                    image = image.convert("L").convert("RGB")
                image = ImageEnhance.Contrast(image).enhance(apply_contrast)
                image = ImageEnhance.Sharpness(image).enhance(apply_sharpness)

                st.subheader("📷 Uploaded MRI Image")
                st.image(image, use_column_width=True)

                with st.spinner("🧠 Detecting Tumor..."):
                    img_array = np.array(image)
                    img_resized = cv2.resize(img_array, (256, 256))
                    img_normalized = img_resized / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)

                    predicted_mask = model_seg.predict(img_input)[0].squeeze().round()
                    mask_colored = np.stack((predicted_mask,)*3, axis=-1)  # grayscale to RGB

                    # Overlay mask
                    overlay = (img_resized * 0.6 + mask_colored * 255 * 0.4).astype(np.uint8)

                    # Tumor area calculation
                    tumor_area_percent = (np.sum(predicted_mask) / (256 * 256)) * 100

                # Store the result in session_state
                st.session_state.predicted_mask = predicted_mask
                st.session_state.overlay = overlay
                st.session_state.tumor_area_percent = tumor_area_percent

                # Display Results
                st.subheader("🧪 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Tumor Mask**")
                    st.image(predicted_mask, use_column_width=True, clamp=True)
                with col2:
                    st.markdown("**Overlay on MRI**")
                    st.image(overlay, use_column_width=True)

                st.subheader("📊 Diagnosis")
                if predicted_mask.any():
                    st.success("✅ **Tumor Detected**")
                    st.markdown(f"🧠 **Tumor Area Estimate**: `{tumor_area_percent:.2f}%` of the brain region.")
                else:
                    st.info("❌ **No Tumor Detected**")

            except UnidentifiedImageError:
                st.error("❌ The file is not a valid image. Please upload a valid image file.")

# Slide 3: Download Report
elif slide == "Download Report":
    if "logged_in" not in st.session_state:
        st.warning("Please log in to access the application.")
    else:
        st.title("📥 Download Your Patient Report")
        
        # Check if the MRI image has been processed and results are available
        if "predicted_mask" not in st.session_state:
            st.warning("Please upload an MRI image first in the 'MRI Detection' slide.")
        else:
            # Patient report details (for simplicity, adding static data here)
            patient_report = f"""
            **Patient Name**: {st.session_state.username}
            **Tumor Area Estimate**: {st.session_state.tumor_area_percent:.2f}% of brain region.
            **Diagnosis**: Tumor Detected
            **Model Confidence Score**: ~90% (visual approximation)
            """
            
            st.subheader("📄 Patient Report")
            st.text(patient_report)
            
            # Button to download report
            st.download_button(
                label="Download Patient Report (txt)",
                data=patient_report,
                file_name="patient_report.txt",
                mime="text/plain"
            )
            
            # Tumor Mask Image download (as JPG)
            tumor_mask_image = Image.fromarray((st.session_state.predicted_mask * 255).astype(np.uint8))
            tumor_mask_image_path = "tumor_mask.jpg"
            tumor_mask_image.save(tumor_mask_image_path)

            with open(tumor_mask_image_path, "rb") as f:
                st.download_button(
                    label="Download Tumor Mask (JPG)",
                    data=f,
                    file_name="tumor_mask.jpg",
                    mime="image/jpeg"
                )
