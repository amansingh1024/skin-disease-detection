import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import os
import time
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#0d1117; --surface:#161b22; --border:#21262d;
    --accent:#58a6ff; --accent2:#3fb950; --danger:#f85149;
    --text:#e6edf3; --muted:#8b949e; --card:#1c2128;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:2rem;}
.hero-title{font-family:'DM Serif Display',serif;font-size:3rem;line-height:1.1;
    background:linear-gradient(135deg,#58a6ff 0%,#3fb950 50%,#f0883e 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.2rem;}
.hero-sub{font-size:1rem;color:var(--muted);font-weight:300;margin-bottom:2rem;}
.metric-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:1.2rem 1.5rem;text-align:center;}
.metric-val{font-family:'DM Serif Display',serif;font-size:2.2rem;color:var(--accent2);}
.metric-label{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:0.2rem;}
.result-box{background:linear-gradient(135deg,#1c2128 0%,#161b22 100%);
    border:1px solid var(--accent2);border-radius:16px;padding:1.8rem 2rem;margin:1rem 0;
    box-shadow:0 0 30px rgba(63,185,80,0.1);}
.result-disease{font-family:'DM Serif Display',serif;font-size:1.9rem;color:var(--accent2);margin-bottom:0.3rem;}
.result-conf{font-size:1rem;color:var(--muted);}
.upload-hint{background:var(--card);border:2px dashed var(--border);border-radius:16px;
    padding:2.5rem;text-align:center;color:var(--muted);font-size:0.95rem;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,#238636,#2ea043)!important;color:white!important;
    border:none!important;border-radius:8px!important;font-weight:500!important;padding:0.6rem 1.5rem!important;}
hr{border-color:var(--border)!important;}
.tag{display:inline-block;background:rgba(88,166,255,0.15);color:var(--accent);
    border:1px solid rgba(88,166,255,0.3);border-radius:20px;padding:0.15rem 0.7rem;
    font-size:0.75rem;font-weight:500;margin-right:0.4rem;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
MODEL_PATH = "best.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── EXACT class names from best.pt (copy-pasted from your checkpoint) ─────────
DEFAULT_CLASSES = [
    "5. Melanocytic Nevi (NV) - 7970",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "2. Melanoma 15.75k",
    "10. Warts Molluscum and other Viral Infections - 2103",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k",
    "1. Eczema 1677",
    "3. Atopic Dermatitis - 1.25k",
]

# ── Clean display names for UI ────────────────────────────────────────────────
DISPLAY_NAMES = {
    "5. Melanocytic Nevi (NV) - 7970"                                 : "Melanocytic Nevi (Moles)",
    "4. Basal Cell Carcinoma (BCC) 3323"                              : "Basal Cell Carcinoma",
    "2. Melanoma 15.75k"                                              : "Melanoma",
    "10. Warts Molluscum and other Viral Infections - 2103"           : "Warts / Viral Infections",
    "6. Benign Keratosis-like Lesions (BKL) 2624"                     : "Benign Keratosis (BKL)",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k"   : "Psoriasis / Lichen Planus",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k"          : "Seborrheic Keratoses",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Tinea / Fungal Infections",
    "1. Eczema 1677"                                                  : "Eczema",
    "3. Atopic Dermatitis - 1.25k"                                    : "Atopic Dermatitis",
}

# ── Disease info ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "2. Melanoma 15.75k": {
        "severity": "High", "color": "#f85149",
        "desc": "A serious form of skin cancer that develops in melanocytes. Early detection is critical.",
        "action": "Seek immediate dermatologist consultation."
    },
    "4. Basal Cell Carcinoma (BCC) 3323": {
        "severity": "High", "color": "#f85149",
        "desc": "Most common skin cancer. Rarely spreads but needs treatment.",
        "action": "Consult a dermatologist for biopsy and treatment."
    },
    "1. Eczema 1677": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Chronic inflammatory skin condition causing itchy, inflamed patches.",
        "action": "Topical creams and lifestyle changes usually effective."
    },
    "3. Atopic Dermatitis - 1.25k": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Common form of eczema. Often begins in childhood.",
        "action": "Moisturizers and avoiding triggers. Consult a doctor if severe."
    },
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": {
        "severity": "Medium", "color": "#f0883e",
        "desc": "Chronic autoimmune condition causing rapid skin cell buildup.",
        "action": "Treatment available. See a dermatologist for management."
    },
    "5. Melanocytic Nevi (NV) - 7970": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Common moles. Usually benign but monitor for ABCDE changes.",
        "action": "Annual skin checks. See a doctor if mole changes shape or colour."
    },
    "6. Benign Keratosis-like Lesions (BKL) 2624": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Non-cancerous growths including seborrheic keratosis and solar lentigo.",
        "action": "Usually no treatment needed. Removal is cosmetic."
    },
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Common benign skin growths. Not cancerous.",
        "action": "No treatment required unless irritated."
    },
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Fungal infection of the skin. Contagious but highly treatable.",
        "action": "Antifungal creams are effective. Keep area clean and dry."
    },
    "10. Warts Molluscum and other Viral Infections - 2103": {
        "severity": "Low", "color": "#3fb950",
        "desc": "Viral skin infections. Usually self-limiting.",
        "action": "Often resolve on their own. Treatment available if persistent."
    },
}

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model     = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, ignore_mismatched_sizes=True)

    # Checkpoint saved a plain Linear(1536, 10) as classifier
    # Keys: classifier.weight [10,1536]  classifier.bias [10]
    model.classifier = nn.Linear(1536, 10)

    if os.path.exists(MODEL_PATH):
        ckpt        = torch.load(MODEL_PATH, map_location=DEVICE)
        state       = ckpt.get("model_state_dict", ckpt)
        class_names = ckpt.get("class_names", DEFAULT_CLASSES)
        model.load_state_dict(state, strict=False)
    else:
        class_names = DEFAULT_CLASSES

    model = model.to(DEVICE)
    model.eval()
    return model, processor, class_names

# ── Predict ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image, model, processor, class_names):
    inputs  = processor(images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
    logits  = model(**inputs).logits
    probs   = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = np.argsort(probs)[::-1]
    top5_names = [DISPLAY_NAMES.get(class_names[i], class_names[i]) for i in top_idx[:5]]
    top5_probs = [float(probs[i]) for i in top_idx[:5]]
    pred_raw   = class_names[top_idx[0]]
    pred_disp  = DISPLAY_NAMES.get(pred_raw, pred_raw)
    return top5_names, top5_probs, pred_raw, pred_disp, float(probs[top_idx[0]])

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.5rem'>🩺</div>
        <div style='font-family:DM Serif Display,serif;font-size:1.3rem;color:#58a6ff'>DermaVision AI</div>
        <div style='font-size:0.75rem;color:#8b949e;margin-top:0.2rem'>v1.0 · DINOv2 Fine-tuned</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    show_info   = st.checkbox("Show disease information", value=True)
    conf_thresh = st.slider("Confidence threshold (%)", 0, 100, 40,
                            help="Warn if top prediction is below this confidence")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class='metric-card' style='margin-bottom:0.7rem'>
        <div class='metric-val'>87.15%</div>
        <div class='metric-label'>Test Accuracy</div>
    </div>
    <div class='metric-card' style='margin-bottom:0.7rem'>
        <div class='metric-val'>0.8728</div>
        <div class='metric-label'>F1 Score</div>
    </div>
    <div class='metric-card'>
        <div class='metric-val'>10</div>
        <div class='metric-label'>Disease Classes</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#8b949e;line-height:1.6'>
    ⚠️ <b>Disclaimer</b><br>
    For educational use only. Not a substitute for professional medical diagnosis.
    Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-title'>DermaVision AI</div>
<div class='hero-sub'>Real-time skin disease detection · DINOv2 · 87.15% accuracy · 10 classes</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Loading AI model..."):
    try:
        model, processor, class_names = load_model()
        if os.path.exists(MODEL_PATH):
            st.success(f"✅ Model loaded — {len(class_names)} classes ready")
        else:
            st.warning(f"⚠️ `best.pt` not found. Place it in the same folder as app.py")
            class_names = DEFAULT_CLASSES
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Upload Skin Image")
    uploaded = st.file_uploader("Choose an image",
                                type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        w, h = image.size
        st.markdown(f"""
        <div style='display:flex;gap:0.5rem;margin-top:0.5rem;flex-wrap:wrap'>
            <span class='tag'>📐 {w}×{h}px</span>
            <span class='tag'>🖼 {uploaded.type}</span>
            <span class='tag'>📦 {uploaded.size//1024} KB</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='upload-hint'>
            <div style='font-size:2.5rem;margin-bottom:0.8rem'>🔬</div>
            <div style='font-size:1rem;color:#e6edf3;margin-bottom:0.3rem'>Drop your skin image here</div>
            <div>Supports JPG, PNG, BMP, WEBP</div>
        </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("### 🔍 Diagnosis Result")
    if uploaded:
        with st.spinner("Analysing image..."):
            t0 = time.time()
            top5_names, top5_probs, pred_raw, pred_disp, pred_conf = predict(
                image, model, processor, class_names)
            elapsed = time.time() - t0

        if pred_conf * 100 < conf_thresh:
            st.warning(f"⚠️ Low confidence ({pred_conf*100:.1f}%). Try a clearer skin photo.")

        info    = DISEASE_INFO.get(pred_raw, {})
        sev     = info.get("severity", "Unknown")
        sev_col = info.get("color", "#58a6ff")

        st.markdown(f"""
        <div class='result-box'>
            <div style='font-size:0.75rem;color:#8b949e;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:0.5rem'>Primary Diagnosis</div>
            <div class='result-disease'>{pred_disp}</div>
            <div class='result-conf'>
                Confidence: <b style='color:#e6edf3'>{pred_conf*100:.2f}%</b>
                &nbsp;·&nbsp; Severity: <b style='color:{sev_col}'>{sev}</b>
                &nbsp;·&nbsp; ⚡ {elapsed*1000:.0f}ms
            </div>
        </div>""", unsafe_allow_html=True)

        if show_info and info:
            with st.expander("📋 About this condition", expanded=True):
                st.markdown(f"**{info.get('desc', '')}**")
                st.info(f"💡 **Recommended action:** {info.get('action', '')}")

        st.markdown("#### Top-5 Predictions")
        colors = ["#3fb950" if i == 0 else "#58a6ff" for i in range(5)]
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in top5_probs],
            y=top5_names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{p*100:.1f}%" for p in top5_probs],
            textposition="outside",
            textfont=dict(size=11, color="#e6edf3"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#21262d",
                       color="#8b949e", title="Confidence (%)"),
            yaxis=dict(color="#e6edf3", autorange="reversed"),
            margin=dict(l=10, r=70, t=10, b=30),
            height=300,
            font=dict(family="DM Sans", color="#e6edf3"),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div style='padding:4rem 2rem;text-align:center;color:#8b949e'>
            <div style='font-size:3rem;margin-bottom:1rem'>🩺</div>
            <div>Upload an image on the left to get an instant diagnosis</div>
        </div>""", unsafe_allow_html=True)

# ── Bottom: class reference ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 Detectable Conditions")
cols = st.columns(5)
for i, raw_cls in enumerate(DEFAULT_CLASSES):
    info     = DISEASE_INFO.get(raw_cls, {})
    sev      = info.get("severity", "Low")
    col      = info.get("color", "#3fb950")
    disp     = DISPLAY_NAMES.get(raw_cls, raw_cls)
    with cols[i % 5]:
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom:0.6rem;text-align:left;padding:0.9rem 1rem'>
            <div style='font-size:0.8rem;font-weight:600;color:#e6edf3;
                        margin-bottom:0.3rem;line-height:1.3'>{disp}</div>
            <div style='font-size:0.7rem;color:{col}'>● {sev} severity</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#8b949e;font-size:0.8rem'>
    DermaVision AI · DINOv2 Fine-tuned · 87.15% accuracy · Educational use only
</div>""", unsafe_allow_html=True)