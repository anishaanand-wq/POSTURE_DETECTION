import json
import time
import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

# Page settings
st.set_page_config(
    page_title="Baby Sleep Posture Detector",
    page_icon="👶",
    layout="centered"
)

# Custom UI styling
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
        }
        .title {
            text-align: center;
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
        }
        .subtitle {
            text-align: center;
            color: #475569;
            font-size: 1.05rem;
            margin-bottom: 25px;
        }
        .result-card {
            background: #ffffff;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
            border: 1px solid #e2e8f0;
        }
        .prediction {
            font-size: 1.7rem;
            font-weight: 800;
            color: #0f172a;
        }
        .confidence {
            color: #2563eb;
            font-weight: 700;
        }
        .bar-container {
            background: #e2e8f0;
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 12px;
        }
        .bar {
            height: 100%;
            border-radius: 999px;
        }
        .disclaimer {
            font-size: 0.8rem;
            color: #64748b;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load class names
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# Load ONNX model only once
@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "baby_sleep_posture_effnetb0.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()
INPUT_NAME = session.get_inputs()[0].name

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = {
    "left": "#3b82f6",
    "prone": "#f97316",
    "sunpine": "#22c55e"
}


def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def predict(image):
    start = time.time()

    input_tensor = preprocess(image)

    logits = session.run(None, {INPUT_NAME: input_tensor})[0][0]

    exp_values = np.exp(logits - np.max(logits))
    probabilities = exp_values / exp_values.sum()

    probabilities = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]
    processing_time = round((time.time() - start) * 1000, 2)

    return prediction, confidence, probabilities, processing_time


# UI
st.markdown('<h1 class="title">👶 Baby Sleep Posture Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a photo of a sleeping baby to detect whether the baby is sleeping left, prone, or supine.</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a baby sleeping image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        if st.button("Analyze Posture", use_container_width=True):
            with st.spinner("Analyzing image..."):
                prediction, confidence, probabilities, processing_time = predict(image)

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="prediction">
                        Predicted posture: {prediction.capitalize()}
                    </p>
                    <p>
                        Confidence: <span class="confidence">
                        {confidence * 100:.1f}%</span>
                    </p>
                    <p style="color:#64748b; font-size:0.9rem;">
                        Processing time: {processing_time} ms
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Posture probabilities")

            for class_name, probability in probabilities.items():
                st.markdown(
                    f"""
                    <div style="margin-bottom:15px;">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            font-weight:600;
                            color:#334155;
                            margin-bottom:5px;
                        ">
                            <span>{class_name.capitalize()}</span>
                            <span>{probability * 100:.1f}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar" style="
                                width:{probability * 100}%;
                                background:{CLASS_COLORS[class_name]};
                            "></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:
    st.info("Please upload a baby sleeping image to begin.")

st.markdown(
    """
    <p class="disclaimer">
        This application is for educational purposes only and should not replace professional medical advice.
    </p>
    """,
    unsafe_allow_html=True
)
