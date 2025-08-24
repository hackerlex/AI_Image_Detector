import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --- Environment setup (silence GPU warnings on Streamlit Cloud) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Model loader with caching ---
@st.cache_resource(show_spinner=False)
def load_model():
    # Load the Keras v3 model saved as .keras
    return keras.models.load_model("model.keras", compile=False)

model = load_model()

# --- Class labels ---
class_labels = ["Real", "AI-generated"]

# --- Preprocessing ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize
    return tf.expand_dims(image, axis=0)        # add batch dimension

# --- Streamlit UI ---
st.title("🖼️ AI Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running model..."):
            input_tensor = preprocess_image(image)
            preds = model.predict(input_tensor)[0]

            # Binary classification
            score = float(preds[0])
            label = class_labels[1] if score > 0.5 else class_labels[0]
            confidence = score if label == class_labels[1] else 1 - score

            st.markdown(f"### Prediction: **{label}** ({confidence:.2%} confidence)")
