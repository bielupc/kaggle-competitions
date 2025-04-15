import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
from PIL import Image
import asyncio

knn = joblib.load('model.joblib')
prediction = [1]

st.title("MNIST Digit Recognizer")
col1, col2= st.columns(2)
with col1:
    st.header("Draw a digit below")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
with col2:
    if canvas_result.image_data is not None:
        st.header("Converted image")
        img = Image.fromarray(canvas_result.image_data, 'RGBA').convert('L')
        img.thumbnail((28, 28), Image.NEAREST)

        img_rescaled = img.convert("RGBA")
        img_rescaled = img_rescaled.resize((280, 280), Image.NEAREST)
        st.image(img_rescaled)
        img = np.asarray(img).flatten()
        prediction = knn.predict([img])
       
st.header(f"Prediction: {prediction[0]}")
