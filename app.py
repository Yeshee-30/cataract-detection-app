import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# Load Model Once (Fast), Cached
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mini_resnet_model.h5")
    return model

model = load_model()

# -----------------------------------------------------------
# Prediction Function
# -----------------------------------------------------------
def predict_image(img):
    img = img.convert("RGB").resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Cataract" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else (1 - pred)
    confidence = round(float(confidence) * 100, 2)
    return label, confidence

# -----------------------------------------------------------
# App UI
# -----------------------------------------------------------

st.set_page_config(
    page_title="Cataract Detection App",
    page_icon="👁️",
    layout="centered",
)

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>
        👁️ Cataract Detection App
    </h1>
    <p style='text-align: center; font-size: 18px;'>
        Upload an eye image, and the AI model will detect whether cataract is present.
    </p>
    """,
    unsafe_allow_html=True
)

# File Upload
uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image")

    with col2:
        with st.spinner("Analyzing the image..."):
            label, confidence = predict_image(img)

        # Result Box
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px;
                        background-color: {'#ffdddd' if label=='Cataract' else '#ddffdd'};
                        text-align: center;
                        border: 1px solid #ccc;">
                <h2 style="color: {'#c0392b' if label=='Cataract' else '#27ae60'};">
                    Prediction: {label}
                </h2>
                <p style="font-size: 18px;">Confidence: {confidence}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence Bar
        st.progress(int(confidence))

        if label == "Cataract":
            st.error("⚠️ Cataract detected. Please consult an eye specialist.")
        else:
            st.success("✔️ No cataract detected.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: grey;'>
        Built with ❤️ using Streamlit and TensorFlow
    </p>
    """,
    unsafe_allow_html=True
)

