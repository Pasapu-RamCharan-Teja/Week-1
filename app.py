import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Garbage Classifier", layout="centered")

st.title("♻️ Garbage Classification System")
st.write("Upload an image and get the predicted waste category.")

@st.cache_resource
def load_my_model():
    return load_model("waste_classifier_model.h5")

model = load_my_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic',
               'trash', 'shoes', 'clothes', 'biological', 'battery']

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.success(f"Predicted class: {pred_class.upper()} (Confidence: {confidence:.2f})")
