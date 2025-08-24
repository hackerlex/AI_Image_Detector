import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load your Keras model once
model = load_model("model.h5")

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize according to your model's training
    return image

def main():
    st.title("AI Image Detector with Custom Model")

    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_img = preprocess_image(image)

        prediction = model.predict(processed_img)

        # Example interpretation for binary classification with sigmoid activation
        confidence = prediction[0][0]
        if confidence > 0.5:
            st.write(f"Likely AI-generated/manipulated with confidence {confidence:.2f}")
        else:
            st.write(f"Likely authentic with confidence {(1 - confidence):.2f}")

if __name__ == "__main__":
    main()
