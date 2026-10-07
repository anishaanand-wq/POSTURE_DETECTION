import json
import time
import io

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Baby Sleep Posture Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 45%, #f0fdf4 100%);
        }

        .hero {
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 55%, #f0fdf4 100%);
            border: 1px solid #dbeafe;
            border-radius: 22px;
            padding: 35px 30px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .hero-badge {
            display: inline-block;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 999px;
            margin-bottom: 15px;
            letter-spacing: 0.4px;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        }

        .hero-subtitle {
            color: #475569;
            font-size: 1.05rem;
            margin-top: 12px;
            margin-bottom: 18px;
        }

        .hero-tags {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .hero-tags span {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 7px 14px;
            border-radius: 999px;
        }

        .result-card {
            background: white;
            border-radius: 20px;
            padding: 25px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        }

        .prediction-text {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
        }

        .confidence-text {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2563eb;
        }

        .bar-label {
            display: flex;
            justify-content: space-between;
            font-weight: 600;
            color: #334155;
            margin-bottom: 4px;
        }

        .bar-bg {
            background: #e2e8f0;
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 14px;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
        }

        .history-card {
            background: white;
            border-radius: 15px;
            padding: 15px;
            border: 1px solid #e2e8f0;
            margin-bottom: 10px;
        }

        .disclaimer {
            text-align: center;
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL CONFIGURATION ----------------

MODEL_PATH = "baby_sleep_posture_effnetb0.onnx"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = {
    "left": "#3b82f6",
    "prone": "#f97316",
    "sunpine": "#22c55e"
}


@st.cache_resource
def load_model():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    return session


with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

session = load_model()
INPUT_NAME = session.get_inputs()[0].name


# ---------------- HELPER FUNCTIONS ----------------

def preprocess_image(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def predict_posture(image):
    start_time = time.time()

    input_tensor = preprocess_image(image)

    logits = session.run(None, {INPUT_NAME: input_tensor})[0][0]

    exp_values = np.exp(logits - np.max(logits))
    probabilities = exp_values / exp_values.sum()

    probabilities = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]
    processing_time = round((time.time() - start_time) * 1000, 2)

    return prediction, confidence, probabilities, processing_time


def show_probability_bars(probabilities):
    for class_name, probability in probabilities.items():
        st.markdown(
            f"""
            <div class="bar-label">
                <span>{class_name.capitalize()}</span>
                <span>{probability * 100:.1f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="
                    width:{probability * 100}%;
                    background:{CLASS_COLORS[class_name]};
                "></div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## Baby Sleep Posture Detector")
    st.markdown(
        """
        Upload or capture a baby sleeping image to detect whether the baby is
        sleeping in a **Left**, **Prone**, or **Supine** posture.
        """
    )

    st.divider()

    st.markdown("### How to Use")
    st.markdown(
        """
        1. Upload an image or take a photo  
        2. Click **Analyze Posture**  
        3. View the predicted posture and confidence  
        4. Download the prediction report if needed  
        """
    )

    st.divider()

    st.markdown("### Model Information")
    st.markdown(
        """
        - **Model:** EfficientNet-B0  
        - **Input Size:** 224 × 224  
        - **Classes:** Left, Prone, Supine  
        - **Deployment:** ONNX Runtime  
        """
    )

    st.divider()

    st.markdown("### Prediction History")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""
                <div class="history-card">
                    <b>{item['prediction'].capitalize()}</b><br>
                    Confidence: {item['confidence'] * 100:.1f}%<br>
                    <span style="color:#64748b; font-size:0.8rem;">
                        {item['time']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No predictions yet.")

# ---------------- MAIN HEADER ----------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-Based Infant Safety Monitoring</div>

        <h1 class="hero-title">Baby Sleep Posture Detector</h1>

        <p class="hero-subtitle">
            Upload or capture a baby sleeping image to identify whether the baby
            is in a <b>Left</b>, <b>Prone</b>, or <b>Supine</b> sleeping posture.
        </p>

        <div class="hero-tags">
            <span>EfficientNet-B0</span>
            <span>ONNX Runtime</span>
            <span>Real-Time Prediction</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUT TABS ----------------

tab1, tab2 = st.tabs(["Upload Image", "Use Camera"])

image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a baby sleeping image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))

with tab2:
    camera_image = st.camera_input("Take a photo of the sleeping baby")

    if camera_image is not None:
        image = Image.open(io.BytesIO(camera_image.getvalue()))

# ---------------- PREDICTION SECTION ----------------

if image is not None:

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.image(
            image,
            caption="Selected baby sleeping image",
            use_container_width=True
        )

        analyze_button = st.button(
            "Analyze Posture",
            use_container_width=True,
            type="primary"
        )

        if st.button("Reset", use_container_width=True):
            st.rerun()

    with right_col:
        if analyze_button:
            with st.spinner("Analyzing baby sleeping posture..."):
                prediction, confidence, probabilities, processing_time = predict_posture(image)

            st.session_state.history.append(
                {
                    "prediction": prediction,
                    "confidence": confidence,
                    "time": time.strftime("%d %b %Y, %I:%M %p")
                }
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="prediction-text">
                        Predicted Posture: {prediction.capitalize()}
                    </p>
                    <p class="confidence-text">
                        Confidence: {confidence * 100:.1f}%
                    </p>
                    <p style="color:#64748b;">
                        Processing time: {processing_time} ms
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Posture Probabilities")
            show_probability_bars(probabilities)

            report = (
                "Baby Sleep Posture Detection Report\n"
                "-----------------------------------\n"
                f"Predicted Posture: {prediction}\n"
                f"Confidence: {confidence * 100:.2f}%\n"
                f"Processing Time: {processing_time} ms\n\n"
                "Class Probabilities:\n"
            )

            for class_name, probability in probabilities.items():
                report += f"{class_name}: {probability * 100:.2f}%\n"

            st.download_button(
                label="Download Prediction Report",
                data=report,
                file_name="baby_sleep_posture_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:
            st.info("Click **Analyze Posture** to get the prediction.")

else:
    st.info("Please upload an image or capture a photo to begin.")

# ---------------- DISCLAIMER ----------------

st.markdown(
    """
    <p class="disclaimer">
        This application is developed for educational and research purposes only.
        It should not be used as a replacement for professional medical advice or baby-monitoring systems.
    </p>
    """,
    unsafe_allow_html=True
)
