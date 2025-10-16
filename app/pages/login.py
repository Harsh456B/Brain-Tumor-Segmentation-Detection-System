import streamlit as st

def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.success("Login Successful!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials. Please try again.")
