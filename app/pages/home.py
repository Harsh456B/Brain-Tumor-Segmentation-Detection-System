import streamlit as st

def home_page():
    st.title("Welcome to Brain Tumor Detection")
    st.subheader("Advanced AI Solutions for Brain Tumor Detection")

    st.markdown(
        """
        This platform uses state-of-the-art AI models to detect brain tumors from MRI scans. 
        Our aim is to assist doctors and healthcare professionals in their diagnostic processes.
        
        - **Accurate Detection**: Our AI model is trained on a large dataset of brain MRI scans to detect abnormalities.
        - **Fast Results**: Upload your MRI scans and get instant results.
        - **Patient Reports**: Download your diagnostic report after the scan.
        
        Explore the menu options to get started.
        """
    )
