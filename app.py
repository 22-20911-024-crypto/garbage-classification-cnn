import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Garbage Classifier",
    page_icon="\u267b\ufe0f",
    layout="centered",
)

# ---------------------------------------------------------
# Load model & class mapping (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("best_model.keras")
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

try:
    model, idx_to_class = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `best_model.keras` and `class_indices.json` "
        "are in the same folder as this app."
    )
    st.stop()

IMG_SIZE = (224, 224)

# Recyclability guidance shown alongside the prediction
RECYCLABLE_INFO = {
    "cardboard": ("\u2705 Recyclable", "Flatten before placing in the recycling bin."),
    "glass":     ("\u2705 Recyclable", "Rinse before recycling; check for local glass-collection rules."),
    "metal":     ("\u2705 Recyclable", "Cans and metal scraps are highly recyclable \u2014 rinse first."),
    "paper":     ("\u2705 Recyclable", "Keep dry and clean for recycling."),
    "plastic":   ("\u26a0\ufe0f Check locally", "Recyclability depends on plastic type \u2014 check the resin code."),
    "trash":     ("\u274c Not recyclable", "This item likely belongs in general waste."),
}

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("\u267b\ufe0f Garbage Classifier")
st.markdown(
    """
Upload a photo of a waste item and this CNN (ResNet50, transfer learning) will classify it
into one of six categories: **cardboard, glass, metal, paper, plastic, or trash** \u2014
helping automate recycling sorting decisions.
"""
)
st.divider()

# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image of a waste item", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("\U0001f50d Classify Waste Item", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            img_resized = img.resize(IMG_SIZE)
            img_array = np.expand_dims(np.array(img_resized).astype("float32"), axis=0)
            img_preprocessed = preprocess_input(img_array)

            preds = model.predict(img_preprocessed, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_class = idx_to_class[pred_idx]
            confidence = float(preds[pred_idx]) * 100

        st.success(f"### Predicted Class: `{pred_class.upper()}`")
        st.metric("Confidence", f"{confidence:.1f}%")

        label, tip = RECYCLABLE_INFO[pred_class]
        st.info(f"**{label}** \u2014 {tip}")

        # Show full probability breakdown
        st.subheader("Class Probabilities")
        prob_dict = {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        st.bar_chart(prob_dict)


