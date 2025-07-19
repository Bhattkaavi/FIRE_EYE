import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import json
import pyttsx3
import base64

# -------------------- CONFIG & LOADING --------------------
@st.cache_data
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change index if female needed
    engine.say(text)
    engine.runAndWait()

def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load assets
rocket = load_lottie_file("assets/rocket.json")
astronaut = load_lottie_file("assets/astronaut.json")
bg = get_base64("assets/starfield.jpg")

# Load YOLOv8 model
model = YOLO("best.pt")

# Set up Streamlit
st.set_page_config(page_title="FIRE-EYE", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"] > .main {{
  background-image: url("data:image/png;base64,{bg}");
  background-size: cover;
  background-attachment: fixed;
}}
html, body, [class*="css"] {{
    font-family: 'Roboto', sans-serif;
    color: #f8fafc;
}}
h1 {{
    font-family: 'Orbitron', sans-serif;
    color: #38bdf8;
    text-align: center;
    font-size: 36px;
}}
.card {{
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 20px;
}}
.result-card {{
    background: rgba(56, 189, 248, 0.1);
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
    border-left: 4px solid #38bdf8;
}}
.alert-card {{
    background-color: rgba(239, 68, 68, 0.15);
    border-left: 5px solid #ef4444;
    padding: 15px;
    border-radius: 10px;
    margin: 8px 0;
}}
.confidence-bar {{
    height: 10px;
    background-color: #1e293b;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 6px;
}}
.confidence-fill {{
    height: 100%;
    background-color: #38bdf8;
    text-align: right;
    padding-right: 5px;
    color: #fff;
    font-size: 10px;
}}
.footer {{
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 40px;
}}
.summary-badge {{
    background-color: #1e3a8a;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st_lottie(rocket, height=120)
with col2:
    st.markdown("<h1>🧯 FIRE-EYE: Astronaut Detection Console</h1>", unsafe_allow_html=True)
with col3:
    st_lottie(astronaut, height=120)

# -------------------- INPUT --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("📤 Upload One or More Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
camera_image = st.camera_input("📷 Or use your Webcam")
voice_enabled = st.toggle("🔊 Enable Voice Summary", value=True)
st.markdown('</div>', unsafe_allow_html=True)

if camera_image:
    uploaded_files = [camera_image]

# -------------------- DETECT BUTTON --------------------
if uploaded_files and st.button("🚀 Run Detection on All"):
    for idx, file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        results = model.predict(source=temp_path, conf=0.25, save=False, show=False)
        r = results[0]
        im = Image.fromarray(r.plot())
        detected_classes = [model.names[int(cls)] for cls in r.boxes.cls] if r.boxes else []
        confidences = [float(conf) for conf in r.boxes.conf] if r.boxes else []
        required = {"FireExtinguisher", "ToolBox", "OxygenTank"}
        detected = set(detected_classes)
        missing = required - detected

        # ---------- DISPLAY RESULT ----------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📸 Detection Result {idx + 1}")
        st.image(im, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- VOICE ALERT ----------
        summary = f"Detected: {', '.join(detected)}."
        if missing:
            summary += f" Missing: {', '.join(missing)}"
        if voice_enabled:
            speak(summary)

        # ---------- Risk Score ----------
        risk_msg = {
            0: ("✅ Area is Safe", "success"),
            1: ("⚠️ Moderate Risk: 1 item missing", "warning"),
            2: ("🔴 High Risk: Multiple items missing!", "error")
        }
        msg, level = risk_msg[min(2, len(missing))]
        getattr(st, level)(msg)

        # ---------- Summary Badge ----------
        st.markdown(f"""
        <div class="summary-badge">
        🛡 <b>Detected:</b> {", ".join(detected) if detected else "None"}<br>
        ❌ <b>Missing:</b> {", ".join(missing) if missing else "None"}
        </div>
        """, unsafe_allow_html=True)

        # ---------- Confidence Bars ----------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🛰 Detection Confidence")
        if detected_classes:
            for cls, conf in zip(detected_classes, confidences):
                percent = int(conf * 100)
                st.markdown(f"""
                <div class="result-card">
                    <b>{cls}</b>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {percent}%;">{percent}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-card'><b>❌ No equipment detected.</b></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Sidebar Log ----------
        with st.sidebar:
            st.subheader("📜 Detection Log")
            st.write(f"🖼 Image {idx+1}: {os.path.basename(temp_path)}")
            for cls in required:
                if cls in detected:
                    st.write(f"✅ {cls}")
                else:
                    st.write(f"❌ {cls}")

        os.remove(temp_path)

# -------------------- FOOTER --------------------
st.markdown('<div class="footer">👨‍🚀 YOLOv8 · Falcon Crew · AlgoVerse 2025 🚀</div>', unsafe_allow_html=True)
