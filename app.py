import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="AI Image Detector", page_icon="🖼️")
st.title("🖼️ AI Image Detector")
st.write("Upload an image to check if it's AI-generated/manipulated or authentic.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

def heuristic_fallback(img: Image.Image):
    """Simple texture-based heuristic for AI vs authentic detection"""
    g = img.convert("L").resize((256,256))
    g_arr = np.array(g, dtype=np.float32) / 255.0
    dx = np.diff(g_arr, axis=1)
    dy = np.diff(g_arr, axis=0)
    texture_var = np.var(dx) + np.var(dy)
    score = float(np.clip((texture_var * 100.0), 0.0, 1.0))
    is_ai = score < 0.15
    prob_ai = 1.0 - score if is_ai else 0.4
    return is_ai, float(np.clip(prob_ai, 0.0, 1.0))

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Always use heuristic (no ML model dependency)
    is_ai, prob_ai = heuristic_fallback(img)
    label = "AI-Generated / Manipulated" if is_ai else "Authentic"

    st.subheader(f"🔍 Result: **{label}**")
    st.caption(f"Confidence: {prob_ai:.2%}")
    st.info("Method: Heuristic fallback (no ML model)")
