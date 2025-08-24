import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# --- Silence Streamlit watchdog errors on some platforms ---
os.environ["STREAMLIT_WATCHDOG_TYPE"] = "polling"

# --- Silence TensorFlow GPU/verbose warnings ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Cache model so it loads only once ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Try direct load
        return keras.models.load_model("model.keras", compile=False)
    except Exception as e:
        st.error(f"⚠️ Could not load model directly: {e}")
        st.info("Rebuilding model architecture (Functional API)...")

        # Fallback: rebuild architecture matching training
        base = keras.applications.MobileNetV2(
            weights=None, include_top=False, input_shape=(224,224,3)
        )
        inputs = keras.Input(shape=(224,224,3))
        x = base(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)

        # Optional: load weights if you separately export them
        if os.path.exists("model.weights.h5"):
            model.load_weights("model.weights.h5")
        return model

model = load_model()

# --- Class labels (edit if needed) ---
class_labels = ["Real", "AI-generated"]

# --- Image preprocessing ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)

# --- Streamlit UI ---
st.title("🖼️ AI Image Detector")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
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

        progress.progress(idx / len(uploaded_files))
