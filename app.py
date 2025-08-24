import os
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Try loading model in multiple formats
@st.cache_resource(show_spinner=False)
def load_model():
    model = None
    try:
        if os.path.exists("model.keras"):
            st.info("Loading model.keras ...")
            model = keras.models.load_model("model.keras", compile=False, safe_mode=False)
        elif os.path.exists("model.h5"):
            st.info("Loading model.h5 ...")
            model = keras.models.load_model("model.h5", compile=False)
        elif os.path.exists("model_saved"):
            st.info("Loading TensorFlow SavedModel (model_saved/) ...")
            model = keras.models.load_model("model_saved", compile=False)
        else:
            st.error("❌ No model file found. Please upload model.keras, model.h5, or model_saved/")
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}")
    return model


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    # Match training preprocessing
    image = image.resize(target_size)
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # same as ImageDataGenerator(rescale=1./255)
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension
    return img_array


# ---------------- Streamlit UI ----------------
st.title("🖼️ AI Image Detector (Local)")

model = load_model()
if model is None:
    st.stop()

class_labels = ["Real", "AI-generated"]

threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)

uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Open & display
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Predict
        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor, verbose=0)
        score = float(preds[0][0])

        # Apply threshold
        label = class_labels[1] if score > threshold else class_labels[0]
        confidence = score if label == class_labels[1] else 1 - score

        st.markdown(f"**Prediction for {uploaded_file.name}: {label} ({confidence:.2%} confidence)**")

        progress.progress(idx / len(uploaded_files))
