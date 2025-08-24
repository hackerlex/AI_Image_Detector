import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Silence Streamlit watchdog errors & TensorFlow warnings
os.environ["STREAMLIT_WATCHDOG_TYPE"] = "polling"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = keras.models.load_model("model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
        st.stop()

model = load_model()

class_labels = ["Real", "AI-generated"]

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    # Resize and normalize the image for MobileNetV2
    image = image.resize(target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Add batch dimension
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

st.title("🖼️ AI Image Detector")

# Add slider UI to adjust classification threshold interactively
threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)

uploaded_files = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        with st.spinner("Analyzing..."):
            input_tensor = preprocess_image(image)
            preds = model.predict(input_tensor, verbose=0)[0][0]

            label = class_labels[1] if preds > threshold else class_labels[0]
            confidence = preds if label == class_labels[1] else 1 - preds

            st.markdown(
                f"**Prediction for {uploaded_file.name}: {label} ({confidence:.2%} confidence)**"
            )

        progress.progress(idx / len(uploaded_files))
