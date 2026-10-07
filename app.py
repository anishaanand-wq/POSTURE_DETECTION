import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Infant Sleep Posture Monitor", page_icon="🌙", layout="wide")


def html(block, target=st):
    """One-line HTML: no blank lines or indentation, so Markdown cannot make a code block."""
    target.markdown("".join(l.strip() for l in block.splitlines()), unsafe_allow_html=True)


html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap');
html, body, .stApp, button, input {font-family:'Manrope',system-ui,'Segoe UI',sans-serif;}
.block-container {padding-top:1.4rem; max-width:1200px;}
.topbar {background:#0B1F33; color:#fff; border-radius:14px; padding:22px 26px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;}
.brand {font-size:1.5rem; font-weight:800; letter-spacing:-0.01em;}
.sub {color:#9fb3c8; font-size:0.92rem; margin-top:2px;}
.chips span {border:1px solid #2d4a66; color:#cfe0ef; border-radius:999px; padding:5px 12px; font-size:0.78rem; font-weight:600; margin-left:6px; display:inline-block;}
.status {border-radius:12px; padding:18px 20px; border:1px solid; border-left-width:6px;}
.status.ok {background:#ecfaf8; border-color:#bfe9e4; border-left-color:#0e8f8a;}
.status.watch {background:#fff4e6; border-color:#fcd9a8; border-left-color:#e07b00;}
.status.unc {background:#f1f5f9; border-color:#d5dde5; border-left-color:#64748b;}
.status .k {font-size:0.82rem; color:#4b5f72; font-weight:600;}
.status .v {font-size:1.75rem; font-weight:800; color:#0b1f33; line-height:1.25;}
.status .d {font-size:0.92rem; color:#4b5f72; margin-top:2px;}
.tiles {display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:14px 0;}
.tile {background:#f4f7fa; border-radius:10px; padding:10px 12px;}
.tile b {display:block; font-size:1.1rem; color:#0b1f33;}
.tile span {font-size:0.76rem; color:#5b6f82;}
.bar-label {display:flex; justify-content:space-between; font-weight:600; font-size:0.92rem; color:#334155; margin-bottom:4px;}
.bar-bg {background:#e2e8f0; height:10px; border-radius:999px; overflow:hidden; margin-bottom:12px;}
.bar-fill {height:100%; border-radius:999px;}
.hist {display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #e2e8f0; font-size:0.88rem; color:#0b1f33;}
.hist small {color:#64748b;}
.stButton > button, .stDownloadButton > button {width:100%; border-radius:10px; font-weight:700;}
[data-testid="stImage"] img {border-radius:10px;}
.foot {text-align:center; color:#64748b; font-size:0.8rem; margin-top:26px;}
</style>
""")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "baby_sleep_posture_effnetb0.onnx"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMG_SIZE, LOW_CONFIDENCE = 224, 0.55
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLORS = {"left": "#3b82f6", "prone": "#e07b00", "sunpine": "#0e8f8a"}
LABELS = {"left": "Left side", "prone": "Stomach (prone)", "sunpine": "Back (supine)"}
label = lambda c: LABELS.get(c, c.capitalize())

for f in (MODEL_PATH, CLASS_NAMES_PATH):
    if not f.exists():
        st.error(f"Missing file: {f.name}. Put it in the same folder as app.py.")
        st.stop()


@st.cache_resource
def load_model():
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


CLASS_NAMES = json.load(open(CLASS_NAMES_PATH))
session = load_model()
INPUT_NAME = session.get_inputs()[0].name


def predict(image):
    t0 = time.time()
    x = (np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0 - MEAN) / STD
    logits = session.run(None, {INPUT_NAME: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    probs = {CLASS_NAMES[i]: float(p[i]) for i in range(len(CLASS_NAMES))}
    top = max(probs, key=probs.get)
    return top, probs[top], probs, round((time.time() - t0) * 1000, 1)


st.session_state.setdefault("history", [])
st.session_state.setdefault("result", None)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### About")
    st.write("Classifies a photo of a sleeping baby as left side, back (supine) or stomach (prone).")
    st.markdown("### Model performance")
    st.caption("Measured on 333 held-out test images.")
    html("""
    <div class="tiles" style="grid-template-columns:1fr 1fr 1fr;">
    <div class="tile"><b>60.1%</b><span>Accuracy</span></div>
    <div class="tile"><b>0.55</b><span>Macro F1</span></div>
    <div class="tile"><b>65.8%</b><span>Prone recall</span></div>
    </div>
    """)
    st.caption("About one in three stomach-sleeping cases is missed, so this is a demonstration, not a monitor.")
    st.markdown("### Recent predictions")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            html(f'<div class="hist"><span><b>{label(h["prediction"])}</b><br><small>{h["time"]}</small></span><span>{h["confidence"]*100:.0f}%</span></div>')
        if st.button("Clear history"):
            st.session_state.history, st.session_state.result = [], None
            st.rerun()
    else:
        st.caption("No predictions yet.")

# ---------------- HEADER ----------------
html("""
<div class="topbar">
<div><div class="brand">Infant Sleep Posture Monitor</div><div class="sub">Upload or capture a photo to classify the sleeping posture.</div></div>
<div class="chips"><span>EfficientNet-B0</span><span>ONNX Runtime</span><span>Research prototype</span></div>
</div>
""")

left, right = st.columns(2, gap="large")

# ---------------- INPUT ----------------
image, data = None, None
with left.container(border=True):
    st.markdown("#### 1. Add an image")
    tab_up, tab_cam = st.tabs(["Upload", "Camera"])
    with tab_up:
        up = st.file_uploader("Baby sleeping image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if up is not None:
            data = up.getvalue()
    with tab_cam:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam is not None:
            data = cam.getvalue()
    if data:
        image = Image.open(io.BytesIO(data))
        st.image(image, caption="Selected image")
        analyze = st.button("Analyze posture", type="primary")
    else:
        analyze = False
        st.info("Choose an image or take a photo to begin.")

key = hashlib.md5(data).hexdigest() if data else None
if analyze and image is not None:
    with st.spinner("Analyzing..."):
        top, conf, probs, ms = predict(image)
    st.session_state.result = dict(key=key, top=top, conf=conf, probs=probs, ms=ms)
    st.session_state.history.append({"prediction": top, "confidence": conf, "time": time.strftime("%d %b, %I:%M %p")})

# ---------------- RESULT ----------------
with right.container(border=True):
    st.markdown("#### 2. Result")
    r = st.session_state.result
    if r and r["key"] == key:
        top, conf = r["top"], r["conf"]
        if conf < LOW_CONFIDENCE:
            cls, head, msg = "unc", "Not sure", "Low confidence. Please check the baby directly."
        elif top == "prone":
            cls, head, msg = "watch", "Stomach (prone)", "Prone position detected. Please check the baby."
        else:
            cls, head, msg = "ok", label(top), "Posture detected."
        html(f'<div class="status {cls}"><div class="k">Predicted posture</div><div class="v">{head}</div><div class="d">{msg}</div></div>')
        html(f'<div class="tiles"><div class="tile"><b>{conf*100:.1f}%</b><span>Confidence</span></div><div class="tile"><b>{r["ms"]} ms</b><span>Processing time</span></div><div class="tile"><b>{len(CLASS_NAMES)}</b><span>Classes</span></div></div>')
        st.markdown("**Probability by posture**")
        for c, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
            html(f'<div class="bar-label"><span>{label(c)}</span><span>{p*100:.1f}%</span></div><div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%;background:{COLORS.get(c, "#64748b")};"></div></div>')
        report = ("Infant Sleep Posture Report\n---------------------------\n"
                  f"Predicted posture: {label(top)}\nConfidence: {conf*100:.2f}%\nProcessing time: {r['ms']} ms\n\n"
                  + "\n".join(f"{label(c)}: {p*100:.2f}%" for c, p in r["probs"].items())
                  + "\n\nResearch prototype. Not a medical or safety device.\n")
        st.download_button("Download report", report, "posture_report.txt", "text/plain")
    else:
        st.info("The prediction will appear here after you click Analyze posture.")

html('<p class="foot">Educational and research use only. This is not a medical device and does not replace watching the baby or professional advice.</p>')
