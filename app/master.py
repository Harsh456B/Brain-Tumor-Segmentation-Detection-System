import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
import os

# Set wide layout and page title
st.set_page_config(page_title="🧠 Brain Tumor Detection | MNC UI", layout="wide")

# Load images
logo_path = "app/images/logo.png"
banner_path = "app/images/company_banner.jpg"

# Make sure images exist
if not os.path.exists(logo_path) or not os.path.exists(banner_path):
    st.error("❌ Logo or Banner image not found. Please check paths.")
    st.stop()

# Load Logo and Banner
logo = Image.open(logo_path)
banner = Image.open(banner_path)

# Display Banner Full Width
st.image(banner, use_column_width=True)
st.markdown("")

# Logo + Title Header
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)
with col2:
    st.markdown("### 🧠 Brain Tumor Detection System - Powered by AI")
    st.caption("An MNC-grade interface for medical diagnosis & imaging")

st.markdown("---")

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_db" not in st.session_state:
    st.session_state.user_db = {}  # basic user db

# --- Authentication ---
if not st.session_state.logged_in:
    selected = option_menu("Welcome", ["Login", "Create Account", "Forgot Password"],
                           icons=["box-arrow-in-right", "person-plus", "key"],
                           orientation="horizontal")

    if selected == "Login":
        st.subheader("🔐 Login to Continue")
        username = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in st.session_state.user_db and st.session_state.user_db[username] == password:
                st.session_state.logged_in = True
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try again.")

    elif selected == "Create Account":
        st.subheader("🆕 Create a New Account")
        new_user = st.text_input("Choose User ID")
        new_pass = st.text_input("Set Password", type="password")
        if st.button("Create"):
            if new_user in st.session_state.user_db:
                st.warning("⚠️ Username already exists.")
            else:
                st.session_state.user_db[new_user] = new_pass
                st.success("✅ Account created! You can now login.")

    elif selected == "Forgot Password":
        st.subheader("🔁 Reset Your Password")
        user_reset = st.text_input("Enter Your Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if user_reset in st.session_state.user_db:
                st.session_state.user_db[user_reset] = new_password
                st.success("✅ Password updated!")
            else:
                st.error("❌ User not found.")
else:
    # --- Main Dashboard ---
    selected = option_menu("Dashboard", ["MRI Detection", "Download Reports", "Logout"],
                           icons=["activity", "cloud-download", "box-arrow-right"],
                           orientation="horizontal")

    if selected == "MRI Detection":
        st.subheader("📥 Upload MRI Scan")
        uploaded_file = st.file_uploader("Upload an MRI Image (JPG/PNG)", type=["jpg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="🧠 Uploaded MRI", width=350)

            # Simulate prediction (Replace with real model)
            st.success("✅ Tumor Detected")
            st.markdown("**Tumor Type**: Glioma Tumor  \n**Confidence**: 94.5%")

            # Save the output
            output_path = os.path.join("app/images", "detected_output.jpg")
            image.save(output_path)

    elif selected == "Download Reports":
        st.subheader("📄 Report Download Section")

        # Report Text
        report = """
--- Brain Tumor Detection Report ---

Patient ID: 54321
Name: John Doe
Tumor Type: Glioma
Accuracy: 94.5%
Date: 22-Apr-2025
        """
        st.code(report)

        # Download Report Text
        b64_txt = base64.b64encode(report.encode()).decode()
        st.markdown(f'<a href="data:file/txt;base64,{b64_txt}" download="report.txt">📄 Download Report</a>', unsafe_allow_html=True)

        # Download MRI Image
        output_image_path = "app/images/detected_output.jpg"
        if os.path.exists(output_image_path):
            with open(output_image_path, "rb") as f:
                img_bytes = f.read()
                b64_img = base64.b64encode(img_bytes).decode()
                st.markdown(f'<a href="data:image/jpg;base64,{b64_img}" download="mri_detected.jpg">🖼️ Download MRI Image</a>', unsafe_allow_html=True)
        else:
            st.info("📌 No image found. Upload one in the 'MRI Detection' section first.")

    elif selected == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()
