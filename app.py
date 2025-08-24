import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --- Silence GPU/TF warnings ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Cache model so it only loads once ---
@st.cache_resource(show_spinner=False)
def load_model():
    return keras.models.load_model("model.keras", compile=False)

model = load_model()

# --- Class labels (edit if training had different order) ---
class_labels = ["Real", "AI-generated"]

# --- Image preprocessing ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize
    return tf.expand_dims(image, axis=0)        # add batch dimension

# --- Streamlit UI ---
st.title("🖼️ AI Image Detector")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        with st.spinner("Analyzing..."):
            input_tensor = preprocess_image(image)
            preds = model.predict(input_tensor, verbose=0)[0]

            score = float(preds[0])
            label = class_labels[1] if score > 0.5 else class_labels[0]
            confidence = score if label == class_labels[1] else 1 - score

            st.markdown(
                f"**Prediction for {uploaded_file.name}: {label} ({confidence:.2%} confidence)**"
            )
