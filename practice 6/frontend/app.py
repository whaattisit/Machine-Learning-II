import os
import requests 
import streamlit as st

API_URL = os.getenv("API_URL", "127.0.0.1:8000")

st.title("Write your review to GTA 6")

review = st.text_area("Review:", height=150, placeholder="What do you think about GTA 6?")

if st.button("Predict"):
    if not review.strip():
        st.warning("Write your review")
    else:
        try:
            r = requests.post(f"{API_URL}/predict/", json={"description" : review}, timeout=10)
            if r.ok:
                st.success(r.json()["msg"])
            else:
                st.error(f"Error {r.status_code}: {r.json().get('detail')}")
        except requests.exceptions.ConnectionError:
            st.error("API isn't available")