import os
import streamlit as st
from PIL import Image
import requests
import json
import numpy as np

MLSERVER_URL = os.environ.get("MLSERVER_URL", "http://localhost:8080")

# Page config
st.set_page_config(
    page_title="AgriTech | Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0c160c; color: #e8f0e8; }

.hero { text-align: center; padding: 4rem 0 3rem 0; }
.hero-icon { font-size: 6rem; line-height: 1; margin-bottom: 1rem; display: block; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 5rem; color: #b8e0b2; margin: 0; letter-spacing: -1px; }
.hero-subtitle { font-size: 1.2rem; color: #7ab87a; margin-top: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; }
.divider { border: none; border-top: 1px solid #1e3a1e; margin: 0 0 2rem 0; }
.section-label { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7ab87a; font-weight: 600; margin-bottom: 0.8rem; }

[data-testid="stFileUploader"] { background-color: #162416 !important; border: 1px solid #2a4a2a !important; border-radius: 12px !important; padding: 1.2rem !important; }
[data-testid="stFileUploader"] section { background-color: #0c160c !important; border: 2px dashed #3a7a3a !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] button { background-color: #4a9e4a !important; color: #ffffff !important; border: none !important; font-size: 1rem !important; font-weight: 600 !important; padding: 0.5rem 1.4rem !important; border-radius: 8px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div span, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] span { color: #dff0df !important; font-size: 0.95rem !important; }
[data-testid="stFileUploader"] small, small { color: #7ab87a !important; font-size: 0.85rem !important; }
[data-testid="stFileUploader"] [data-testid="stTooltipIcon"] { display: none !important; }

.result-card { background: #162416; border: 1px solid #2a4a2a; border-radius: 12px; padding: 2rem 2.5rem; margin-top: 1.2rem; }
.result-label { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.14em; color: #7ab87a; margin-bottom: 0.3rem; font-weight: 600; }
.result-disease { font-family: 'DM Serif Display', serif !important; font-size: 2.8rem !important; font-weight: 400 !important; color: #b8e0b2; margin: 0 0 1.5rem 0; }
.confidence-row { display: flex; align-items: center; gap: 1rem; margin-top: 0.3rem; margin-bottom: 1.5rem; }
.confidence-value { font-size: 2rem; font-weight: 600; color: #dff0df; min-width: 80px; }
.confidence-bar-bg { flex: 1; background: #0c160c; border-radius: 999px; height: 12px; overflow: hidden; }
.confidence-bar-fill { height: 100%; border-radius: 999px; }

.top3-title { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.14em; color: #7ab87a; font-weight: 600; margin-bottom: 1rem; border-top: 1px solid #2a4a2a; padding-top: 1.2rem; }
.top3-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.9rem; }
.top3-name { font-size: 1.1rem; color: #c8e6c8; min-width: 220px; }
.top3-bar-bg { flex: 1; background: #0c160c; border-radius: 999px; height: 8px; overflow: hidden; }
.top3-bar-fill { height: 100%; border-radius: 999px; }
.top3-pct { font-size: 1.1rem; color: #7ab87a; min-width: 55px; text-align: right; }

.footer { text-align: center; color: #5a8a5a; font-size: 0.9rem; margin-top: 2.5rem; padding-bottom: 1.5rem; letter-spacing: 0.05em; }
div[data-baseweb="tooltip"], div[role="tooltip"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <span class="hero-icon">🌿</span>
    <h1 class="hero-title">AgriTech</h1>
    <p class="hero-subtitle">Plant Disease Classifier</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# Upload
st.markdown('<p class="section-label">Upload a Leaf Photo</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"]
)

if uploaded_file:
    # verify() consumes the stream, so we seek back and re-open for display and inference
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    except Exception:
        st.error("Invalid file. Please upload a valid image.")
        st.stop()

    with st.spinner("Analysing..."):
        try:
            # Build the V2 inference request manually (equivalent to NumpyCodec.encode_input)
            # mlserver cannot be imported on Windows due to signal.SIGQUIT, so we construct the dict directly
            img_array = np.array(image)
            h, w, c = img_array.shape
            response = requests.post(
                MLSERVER_URL + "/v2/models/plant-disease-classifier/infer",
                json={
                    "inputs": [{
                        "name": "payload",
                        "shape": [h, w, c],
                        "datatype": "UINT8",
                        "data": img_array.flatten().tolist()
                    }]
                }
            )
            raw = response.json()
            if "outputs" not in raw:
                st.error(f"Could not connect to the API. Make sure the backend is running.")
                st.stop()
            result = json.loads(raw["outputs"][0]["data"][0])

            disease = result["disease"].replace("___", " — ").replace("_", " ")
            confidence = result["confidence"]
            top3 = result["top3"]

            # Bar color based on confidence
            if confidence >= 80:
                bar_color = "linear-gradient(90deg, #4a9e4a, #a8d5a2)"
            elif confidence >= 50:
                bar_color = "linear-gradient(90deg, #9e8a4a, #d5c8a2)"
            else:
                bar_color = "linear-gradient(90deg, #9e4a4a, #d5a2a2)"

            top3_rows = ""
            for item in top3:
                name = item["disease"].replace("___", " — ").replace("_", " ")
                pct = item["confidence"]
                if pct >= 80:
                    top3_bar_color = "#4a9e4a"
                elif pct >= 50:
                    top3_bar_color = "#9e8a4a"
                else:
                    top3_bar_color = "#9e4a4a"
                top3_rows += (
                    '<div class="top3-row">'
                    '<span class="top3-name">' + name + '</span>'
                    '<div class="top3-bar-bg">'
                    '<div class="top3-bar-fill" style="width:' + str(pct) + '%;background:' + top3_bar_color + ';"></div>'
                    '</div>'
                    '<span class="top3-pct">' + str(pct) + '%</span>'
                    '</div>'
                )

            result_html = (
                '<div class="result-card">'
                '<p class="result-label">Detected Disease</p>'
                '<p class="result-disease">' + disease + '</p>'
                '<p class="result-label">Confidence Score</p>'
                '<div class="confidence-row">'
                '<span class="confidence-value">' + str(confidence) + '%</span>'
                '<div class="confidence-bar-bg">'
                '<div class="confidence-bar-fill" style="width:' + str(confidence) + '%;background:' + bar_color + ';"></div>'
                '</div>'
                '</div>'
                '<p class="top3-title">Other Possibilities</p>'
                + top3_rows +
                '</div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)

        except Exception as e:
            st.error("Could not connect to the API. Make sure the backend is running.")

# Footer
st.markdown('<p class="footer">AgriTech · Plant Disease Detection · Prototype</p>', unsafe_allow_html=True)
