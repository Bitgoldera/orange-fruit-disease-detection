from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "orange_disease_efficientnet_b0_final.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
GATE_MODEL_PATH = ROOT / "models" / "orange_gate_final.keras"
GATE_THRESHOLD_PATH = ROOT / "models" / "orange_gate_threshold.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Orange Disease Detector",
    page_icon="🍊",
    layout="wide",
)


st.markdown('\n<style>\n.stApp {\n    background: radial-gradient(circle at 95% 3%, #fff0d8, transparent 25rem),\n                linear-gradient(180deg, #fbf9f2, #f5f7f1);\n    color: #203629;\n}\n.block-container {\n    max-width: 1200px !important;\n    padding-top: 1.5rem !important;\n    padding-bottom: 3rem !important;\n    padding-left: 1.5rem !important;\n    padding-right: 1.5rem !important;\n}\n[data-testid="stHeader"] { background: transparent; }\nsection[data-testid="stSidebar"] {\n    background: #eff4eb;\n    border-right: 1px solid #dfe8db;\n}\n.hero {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    gap: 1rem;\n    padding: clamp(1.4rem, 4vw, 2.6rem);\n    margin: .3rem 0 1.5rem;\n    border-radius: 24px;\n    color: white;\n    background: linear-gradient(120deg, #153927, #28583b 72%, #397147);\n    box-shadow: 0 15px 38px #244c3519;\n}\n.hero-copy { max-width: 760px; }\n.hero-kicker {\n    color: #ffcf91;\n    font-size: .75rem;\n    font-weight: 800;\n    letter-spacing: .13em;\n    text-transform: uppercase;\n}\n.hero-title {\n    margin: .65rem 0;\n    color: white;\n    font-size: clamp(2rem, 4vw, 3.1rem);\n    font-weight: 850;\n    letter-spacing: -.04em;\n    line-height: 1.08;\n}\n.hero-desc { color: #e2eee2; line-height: 1.6; }\n.hero-tags { display: flex; flex-wrap: wrap; gap: .5rem; }\n.hero-tag {\n    padding: .4rem .7rem;\n    border: 1px solid #ffffff30;\n    border-radius: 9px;\n    background: #ffffff12;\n    color: #f4f7ee;\n    font-size: .78rem;\n}\n.hero-fruit {\n    display: grid;\n    place-items: center;\n    flex-shrink: 0;\n    width: clamp(75px, 12vw, 140px);\n    height: clamp(75px, 12vw, 140px);\n    border: 1px solid #ffffff30;\n    border-radius: 50%;\n    background: #ffffff12;\n    font-size: clamp(3rem, 7vw, 5rem);\n}\n.section-kicker {\n    color: #73816f;\n    font-size: .72rem;\n    font-weight: 850;\n    letter-spacing: .13em;\n    text-transform: uppercase;\n}\n.section-title {\n    color: #214b35;\n    font-size: 1.35rem;\n    font-weight: 800;\n}\n[data-testid="stFileUploader"] {\n    padding: .35rem;\n    border: 1.5px dashed #bed2bd;\n    border-radius: 16px;\n    background: #ffffffd9;\n}\n[data-testid="stImage"] img {\n    max-width: 100%;\n    object-fit: contain;\n    border-radius: 15px;\n}\n.stButton > button {\n    min-height: 2.8rem;\n    border-radius: 11px;\n    font-weight: 750;\n}\n.stButton > button[kind="primary"] {\n    color: white !important;\n    border: 1px solid #df791d !important;\n    background: linear-gradient(100deg, #f39730, #e77518) !important;\n    box-shadow: 0 6px 16px #e8792025;\n}\n[data-testid="stMetric"] {\n    padding: .9rem 1rem;\n    border: 1px solid #e5ebdf;\n    border-radius: 14px;\n    background: white;\n}\n[data-testid="stMetricValue"] { color: #214b35 !important; }\n[data-testid="stProgress"] > div > div {\n    background: linear-gradient(90deg, #f6ad45, #e7771e);\n}\n[data-testid="stAlert"] { border-radius: 12px; }\nh1, h2, h3 { color: #203629; }\n@media (max-width: 650px) {\n    .block-container {\n        padding-left: 1rem !important;\n        padding-right: 1rem !important;\n    }\n    .hero { padding: 1.15rem; border-radius: 18px; }\n    .hero-fruit { width: 65px; height: 65px; font-size: 2.4rem; }\n}\n\n\n/* Premium sidebar */\nsection[data-testid="stSidebar"] .block-container {\n  padding-top: 1.35rem !important;\n  padding-bottom: 1.5rem !important;\n}\n.side-brand {\n  display: flex;\n  align-items: center;\n  gap: .75rem;\n  margin-bottom: .75rem;\n}\n.side-logo {\n  display: grid;\n  place-items: center;\n  flex: 0 0 48px;\n  width: 48px;\n  height: 48px;\n  border-radius: 15px;\n  background: linear-gradient(145deg, #FFE6B8, #F7A83B);\n  box-shadow: 0 5px 12px #D96C1223;\n  font-size: 1.65rem;\n}\n.side-brand-name {\n  color: #214B35;\n  font-size: 1.28rem;\n  font-weight: 850;\n  letter-spacing: -.035em;\n}\n.side-brand-sub {\n  margin-top: 2px;\n  color: #687D6D;\n  font-size: .64rem;\n  font-weight: 800;\n  letter-spacing: .11em;\n}\n.side-intro {\n  margin: .4rem 0 1rem;\n  color: #526858;\n  font-size: .84rem;\n  line-height: 1.6;\n}\n.side-rule {\n  height: 1px;\n  margin: 1rem 0;\n  background: #D8E4D5;\n}\n.side-label {\n  margin-bottom: .65rem;\n  color: #6A7E6C;\n  font-size: .68rem;\n  font-weight: 850;\n  letter-spacing: .12em;\n}\n.side-step {\n  display: flex;\n  align-items: flex-start;\n  gap: .7rem;\n  margin: .42rem 0;\n  padding: .58rem .6rem;\n  border: 1px solid #E1E9DE;\n  border-radius: 12px;\n  background: #F9FBF7;\n}\n.side-step-number {\n  display: grid;\n  place-items: center;\n  flex: 0 0 29px;\n  width: 29px;\n  height: 29px;\n  border-radius: 9px;\n  color: #214B35;\n  background: #E5F0E2;\n  font-size: .72rem;\n  font-weight: 850;\n}\n.side-step-title {\n  color: #294735;\n  font-size: .81rem;\n  font-weight: 780;\n}\n.side-step-copy {\n  margin-top: 2px;\n  color: #6A7C6D;\n  font-size: .7rem;\n  line-height: 1.4;\n}\n.side-chip-list {\n  display: flex;\n  flex-wrap: wrap;\n  gap: .38rem;\n}\n.side-chip {\n  padding: .36rem .52rem;\n  border: 1px solid #DCE7D9;\n  border-radius: 8px;\n  color: #385740;\n  background: #F9FBF7;\n  font-size: .68rem;\n  font-weight: 650;\n}\n\n/* Team card */\n.side-team-card {\n  margin: .35rem 0 1rem;\n  padding: .8rem;\n  border: 1px solid #D9E5D5;\n  border-radius: 15px;\n  background: #F9FBF7;\n  box-shadow: 0 4px 12px #214B3508;\n}\n.side-team-heading {\n  display: flex;\n  align-items: center;\n  gap: .65rem;\n  margin-bottom: .7rem;\n}\n.side-team-icon {\n  display: grid;\n  place-items: center;\n  flex: 0 0 38px;\n  width: 38px;\n  height: 38px;\n  border-radius: 12px;\n  color: #FFFFFF;\n  background: linear-gradient(145deg, #214B35, #397147);\n  font-size: 1.1rem;\n}\n.side-team-title {\n  color: #214B35;\n  font-size: .91rem;\n  font-weight: 850;\n}\n.side-team-subtitle {\n  margin-top: 2px;\n  color: #687B6B;\n  font-size: .66rem;\n  line-height: 1.4;\n}\n.side-member-row {\n  display: flex;\n  align-items: center;\n  gap: .62rem;\n  padding: .38rem .1rem;\n}\n.side-member-avatar {\n  display: grid;\n  place-items: center;\n  flex: 0 0 29px;\n  width: 29px;\n  height: 29px;\n  border: 1px solid #D7E5D3;\n  border-radius: 9px;\n  color: #214B35;\n  background: #EAF2E7;\n  font-size: .65rem;\n  font-weight: 850;\n}\n.side-member-name {\n  color: #344A39;\n  font-size: .79rem;\n  font-weight: 650;\n}\n.side-github-link {\n  display: flex;\n  align-items: center;\n  justify-content: space-between;\n  gap: .5rem;\n  margin-top: .65rem;\n  padding: .68rem .75rem;\n  border: 1px solid #214B35;\n  border-radius: 10px;\n  color: #FFFFFF !important;\n  background: #214B35;\n  font-size: .77rem;\n  font-weight: 750;\n  text-decoration: none !important;\n  transition: background .18s ease, transform .18s ease;\n}\n.side-github-link:hover {\n  color: #FFFFFF !important;\n  background: #326544;\n  transform: translateY(-1px);\n}\n.side-github-link span {\n  color: #FFFFFF !important;\n}\n\n.side-notice {\n  margin-top: 1.1rem;\n  padding: .8rem .85rem;\n  border: 1px solid #E8D3AE;\n  border-radius: 12px;\n  background: #FFF5E4;\n  color: #60471F;\n  font-size: .74rem;\n  line-height: 1.55;\n}\n.side-notice strong {\n  display: block;\n  margin-bottom: .2rem;\n  color: #754D19;\n  font-size: .78rem;\n}\n\n/* Less empty space between the upload heading and drop area */\n.section-kicker { margin-bottom: .1rem !important; }\n.section-title { margin-bottom: .25rem !important; }\ndiv[data-testid="stFileUploader"] {\n  margin-top: -.45rem !important;\n}\n\n/* Reliable contrast for upload text and filename */\n[data-testid="stFileUploaderDropzone"] {\n  background: #FFFCF6 !important;\n  border-color: #D7B47A !important;\n}\n[data-testid="stFileUploaderDropzone"] p,\n[data-testid="stFileUploaderDropzone"] span,\n[data-testid="stFileUploaderDropzone"] small {\n  color: #344A39 !important;\n  -webkit-text-fill-color: #344A39 !important;\n  opacity: 1 !important;\n}\n[data-testid="stFileUploader"] button {\n  background: #FFF0D8 !important;\n  color: #75420F !important;\n  -webkit-text-fill-color: #75420F !important;\n}\n\n/* Keep the result tabs readable without hovering */\n.stTabs [role="tab"] {\n  color: #214B35 !important;\n  -webkit-text-fill-color: #214B35 !important;\n  opacity: 1 !important;\n}\n.stTabs [role="tab"][aria-selected="true"] {\n  background: #214B35 !important;\n  color: #FFFFFF !important;\n  -webkit-text-fill-color: #FFFFFF !important;\n}\n.stTabs [role="tab"][aria-selected="true"] * {\n  color: #FFFFFF !important;\n  -webkit-text-fill-color: #FFFFFF !important;\n}\n[data-testid="stAlert"] p,\n[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {\n  color: #26382D !important;\n  -webkit-text-fill-color: #26382D !important;\n  opacity: 1 !important;\n}\n@media (prefers-reduced-motion: reduce) {\n  .hero, .hero-fruit, .hero-tag, .side-step,\n  [data-testid="stMetric"], [data-testid="stImage"] {\n    animation: none !important;\n    transition: none !important;\n  }\n}\n\n</style>\n', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-copy">
    <div class="hero-kicker">🍃 Smart Citrus Health</div>
    <div class="hero-title">Orange Fruit<br>Disease Detector</div>
    <div class="hero-desc">
      AI-assisted citrus image analysis with disease scores
      and a visual explanation.
    </div>
    <div class="hero-tags">
      <span class="hero-tag">EfficientNet-B0</span>
      <span class="hero-tag">5 disease classes</span>
      <span class="hero-tag">Grad-CAM</span>
    </div>
  </div>
  <div class="hero-fruit">🍊</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="side-brand">
  <div class="side-logo">🍊</div>
  <div>
    <div class="side-brand-name">CitrusCare</div>
    <div class="side-brand-sub">SMART CITRUS HEALTH</div>
  </div>
</div>

<div class="side-intro">
  AI-assisted screening for possible orange fruit diseases.
  Upload a clear image to begin your analysis.
</div>

<div class="side-rule"></div>
<div class="side-label">HOW IT WORKS</div>

<div class="side-step">
  <div class="side-step-number">01</div>
  <div>
    <div class="side-step-title">Upload image</div>
    <div class="side-step-copy">Choose a clear fruit photo</div>
  </div>
</div>
<div class="side-step">
  <div class="side-step-number">02</div>
  <div>
    <div class="side-step-title">Validate input</div>
    <div class="side-step-copy">Check the orange gate result</div>
  </div>
</div>
<div class="side-step">
  <div class="side-step-number">03</div>
  <div>
    <div class="side-step-title">Predict condition</div>
    <div class="side-step-copy">Compare the five class scores</div>
  </div>
</div>
<div class="side-step">
  <div class="side-step-number">04</div>
  <div>
    <div class="side-step-title">Explore Grad-CAM</div>
    <div class="side-step-copy">Review the model's visual explanation</div>
  </div>
</div>

<div class="side-rule"></div>
<div class="side-label">SUPPORTED CLASSES</div>
<div class="side-chip-list">
  <span class="side-chip">Black Spot</span>
  <span class="side-chip">Canker</span>
  <span class="side-chip">Greening</span>
  <span class="side-chip">Healthy</span>
  <span class="side-chip">Melanose</span>
</div>


<div class="side-rule"></div>
<div class="side-label">PROJECT TEAM</div>
<div class="side-team-card">
  <div class="side-team-heading">
    <div class="side-team-icon">✦</div>
    <div>
      <div class="side-team-title">Meet the Team</div>
      <div class="side-team-subtitle">Orange Fruit Disease Detection</div>
    </div>
  </div>

  <div class="side-member-row">
    <span class="side-member-avatar">HP</span>
    <span class="side-member-name">Harshit Patidar</span>
  </div>
  <div class="side-member-row">
    <span class="side-member-avatar">NK</span>
    <span class="side-member-name">Nikhil Kumar</span>
  </div>
  <div class="side-member-row">
    <span class="side-member-avatar">SK</span>
    <span class="side-member-name">Sumit Kumar</span>
  </div>
  <div class="side-member-row">
    <span class="side-member-avatar">PS</span>
    <span class="side-member-name">Parth Sarthi</span>
  </div>

  <a class="side-github-link"
     href="https://github.com/HarshitPatidarOfficial/orange-fruit-disease-detection"
     target="_blank" rel="noopener noreferrer">
     <span>View project on GitHub</span><span>↗</span>
  </a>
</div>

<div class="side-notice">
  <strong>Educational prototype</strong>
  Model predictions are estimates, not confirmed agricultural diagnoses.
</div>
""", unsafe_allow_html=True)



@st.cache_resource
def load_resources():
    model = keras.models.load_model(MODEL_PATH)
    gate_model = keras.models.load_model(GATE_MODEL_PATH)

    with GATE_THRESHOLD_PATH.open(encoding="utf-8") as f:
        gate_config = json.load(f)

    with CLASS_PATH.open(encoding="utf-8") as f:
        classes = json.load(f)

    # Access the trained EfficientNet backbone.
    backbone = model.get_layer("efficientnetb0")

    # Create a feature extractor ending at the final convolution.
    conv_extractor = keras.Model(
        inputs=backbone.input,
        outputs=backbone.get_layer("top_conv").output,
    )

    # Reconstruct the forward path so Grad-CAM can calculate
    # gradients from the predicted class to the final convolution.
    inputs = keras.Input(shape=(224, 224, 3))

    # Apply the three augmentation layers individually.
    x = model.get_layer("random_flip")(inputs, training=False)
    x = model.get_layer("random_rotation")(x, training=False)
    x = model.get_layer("random_zoom")(x, training=False)

    conv_outputs = conv_extractor(x, training=False)

    # EfficientNet's final convolutional activation.
    x = backbone.get_layer("top_bn")(conv_outputs)
    x = backbone.get_layer("top_activation")(x)

    # The pooling and classifier layers belong to the outer model.
    # The saved EfficientNet backbone contains its own avg_pool layer.
    try:
        pooling_layer = backbone.get_layer("avg_pool")
    except ValueError:
        # Fallback for architectures that pool outside the backbone.
        pooling_layer = next(
            (
                layer for layer in model.layers
                if isinstance(layer, keras.layers.GlobalAveragePooling2D)
            ),
            None,
        )

    if pooling_layer is None:
        raise ValueError(
            "Could not locate the global average pooling layer "
            "for Grad-CAM."
        )
    dropout_layer = next(
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Dropout)
    )
    classifier_layer = next(
        layer for layer in model.layers
        if isinstance(layer, keras.layers.Dense)
        and layer.units == len(classes)
    )

    x = pooling_layer(x)
    x = dropout_layer(x, training=False)
    predictions = classifier_layer(x)

    grad_model = keras.Model(
        inputs=inputs,
        outputs=[conv_outputs, predictions],
    )

    return model, grad_model, classes, gate_model, gate_config


def create_gradcam(grad_model, image_batch, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            image_batch,
            training=False,
        )
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients.")

    # Average gradients across the spatial dimensions.
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

    activations = conv_outputs[0]
    heatmap = tf.reduce_sum(
        activations * weights,
        axis=-1,
    )

    # Keep positive contributions and normalize.
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + keras.backend.epsilon()
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
    )

    return heatmap.numpy().squeeze()


st.markdown('<div class="section-kicker">STEP 01 · IMAGE INPUT</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload your image</div>', unsafe_allow_html=True)





st.markdown('\n<style>\n/* Disease score measurement bars */\n.score-track {\n    display: block !important;\n    box-sizing: border-box !important;\n    width: 100% !important;\n    height: 12px !important;\n    margin: 9px 0 17px !important;\n    overflow: hidden !important;\n    background: #E4EBE1 !important;\n    border: 1px solid #CAD9C7 !important;\n    border-radius: 999px !important;\n}\n.score-fill {\n    display: block !important;\n    box-sizing: border-box !important;\n    height: 100% !important;\n    min-width: 0 !important;\n    max-width: 100% !important;\n    border-radius: 999px !important;\n    background: linear-gradient(90deg, #F6B347, #F28C28, #D96C12) !important;\n    transition: width .8s ease !important;\n}\n@media (prefers-reduced-motion: reduce) {\n    .score-fill { transition: none !important; }\n}\n</style>\n', unsafe_allow_html=True)

st.markdown('\n<style>\n/* Tabs: readable by default, clear hover, strong selected state */\n.stTabs [role="tab"] {\n    background: #EEF4EB !important;\n    border: 1px solid #D8E4D5 !important;\n    border-radius: 10px 10px 0 0 !important;\n    color: #214B35 !important;\n    -webkit-text-fill-color: #214B35 !important;\n    opacity: 1 !important;\n    font-weight: 750 !important;\n}\n.stTabs [role="tab"] *,\n.stTabs [role="tab"] span {\n    color: #214B35 !important;\n    -webkit-text-fill-color: #214B35 !important;\n    opacity: 1 !important;\n}\n.stTabs [role="tab"]:hover:not([aria-selected="true"]) {\n    background: #FFF0D8 !important;\n    border-color: #E9B36A !important;\n}\n.stTabs [role="tab"][aria-selected="true"] {\n    background: #214B35 !important;\n    border-color: #214B35 !important;\n    box-shadow: inset 0 -3px #F28C28 !important;\n    color: #FFFFFF !important;\n    -webkit-text-fill-color: #FFFFFF !important;\n}\n.stTabs [role="tab"][aria-selected="true"] *,\n.stTabs [role="tab"][aria-selected="true"] span {\n    color: #FFFFFF !important;\n    -webkit-text-fill-color: #FFFFFF !important;\n}\n\n/* Sage track; green leading prediction; citrus-orange other classes */\n.score-track {\n    display: block !important;\n    box-sizing: border-box !important;\n    width: 100% !important;\n    height: 11px !important;\n    margin: 8px 0 16px !important;\n    overflow: hidden !important;\n    background: #E7EEE4 !important;\n    border: 1px solid #D3E0D0 !important;\n    border-radius: 999px !important;\n}\n.score-fill-top,\n.score-fill-other {\n    display: block !important;\n    height: 100% !important;\n    max-width: 100% !important;\n    border-radius: 999px !important;\n    transition: width .7s ease !important;\n}\n.score-fill-top {\n    background: linear-gradient(90deg, #214B35, #397147) !important;\n}\n.score-fill-other {\n    background: linear-gradient(90deg, #F6B94F, #F28C28, #D96C12) !important;\n}\n@media (prefers-reduced-motion: reduce) {\n    .score-fill-top, .score-fill-other {\n        transition: none !important;\n    }\n}\n</style>\n', unsafe_allow_html=True)


def get_treatment_advice(class_name):
    """General educational guidance, not a confirmed diagnosis."""
    key = str(class_name).lower().replace("_", " ").replace("-", " ").strip()
    guides = {
        "black spot": (
            "The model flagged possible citrus black spot. Image predictions "
            "cannot confirm the pathogen.",
            [
                "Ask a local horticulture officer or Krishi Vigyan Kendra (KVK) "
                "to verify the symptoms.",
                "Follow local guidance for fallen fruit and affected plant "
                "material; do not move suspect material between orchards.",
                "Maintain suitable canopy airflow and avoid prolonged foliage "
                "wetness where practical.",
                "Ask your local extension service about currently approved "
                "management measures and timing.",
            ],
        ),
        "canker": (
            "The result resembles citrus canker, a bacterial disease. Confirm "
            "the cause before choosing control measures.",
            [
                "Inspect leaves, twigs and fruit for raised lesions, and ask "
                "an expert to verify the symptoms.",
                "Where locally recommended, remove affected twigs in dry "
                "conditions and dispose of material safely.",
                "Clean pruning tools between trees and avoid moving suspect "
                "plant material to other orchards.",
                "Ask your KVK or state horticulture department about "
                "locally registered control options.",
            ],
        ),
        "greening": (
            "The model flagged possible citrus greening (Huanglongbing). "
            "Seek prompt expert confirmation; similar symptoms can have "
            "other causes.",
            [
                "Contact a horticulture officer or plant diagnostic service "
                "promptly to confirm the disease.",
                "Use certified, disease-free citrus planting material for "
                "new plantings and replacements.",
                "Ask local extension staff about current, approved management "
                "of the Asian citrus psyllid vector.",
                "Do not move budwood or planting material from a suspect tree; "
                "ask an expert how to handle a confirmed infected tree.",
            ],
        ),
        "melanose": (
            "The result resembles citrus melanose. Surface blemishes alone "
            "cannot confirm the disease.",
            [
                "Have the symptoms verified, especially if fruit quality "
                "or marketability is affected.",
                "Follow local sanitation advice for dead twigs and dead wood.",
                "Maintain canopy airflow and avoid prolonged wetness where "
                "practical.",
                "Ask a local horticulture service about suitable management "
                "timing and currently registered products.",
            ],
        ),
        "healthy": (
            "The model's top category is Healthy. This does not prove that "
            "the tree or fruit is disease-free.",
            [
                "Continue checking fruit, leaves, shoots and branches.",
                "Maintain orchard sanitation, appropriate irrigation and "
                "balanced nutrition.",
                "Use certified disease-free planting material for new plants.",
                "If symptoms persist or spread, seek expert assessment even "
                "if the model predicts Healthy.",
            ],
        ),
    }
    summary, actions = guides.get(key, (
        "The prediction needs expert confirmation before choosing a "
        "disease-management action.",
        [
            "Capture clear images of several affected plant parts.",
            "Consult a local horticulture officer or KVK.",
            "Do not apply pesticides based only on this model output.",
        ],
    ))
    return {"summary": summary, "actions": actions}


def create_treatment_pdf(
    filename, predicted_class, score, orange_probability, threshold,
    class_scores, advice, original, overlay,
):
    """Generate a PDF report in memory."""
    from io import BytesIO
    from datetime import datetime
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as PDFImage,
    )

    forest = colors.HexColor("#214B35")
    orange = colors.HexColor("#F28C28")
    ivory = colors.HexColor("#FBF8F1")
    ink = colors.HexColor("#26382D")
    muted = colors.HexColor("#586B5C")

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="CitrusCare Disease Analysis Report",
        author="CitrusCare",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CC_Title", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=forest, alignment=1, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="CC_Sub", parent=styles["Normal"],
        fontSize=8.5, leading=12, textColor=muted,
        alignment=1, spaceAfter=15,
    ))
    styles.add(ParagraphStyle(
        name="CC_Section", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=12,
        textColor=forest, spaceBefore=12, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CC_Body", parent=styles["BodyText"],
        fontSize=8.7, leading=12, textColor=ink, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="CC_Small", parent=styles["BodyText"],
        fontSize=7.4, leading=10, textColor=muted, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CC_Head", parent=styles["BodyText"],
        fontName="Helvetica-Bold", fontSize=8.5,
        textColor=colors.white,
    ))

    story = [
        Paragraph("CitrusCare", styles["CC_Title"]),
        Paragraph(
            "ORANGE FRUIT DISEASE DETECTION · ANALYSIS REPORT",
            styles["CC_Sub"],
        ),
        Paragraph("Prediction summary", styles["CC_Section"]),
    ]

    rows = [[
        Paragraph("Report detail", styles["CC_Head"]),
        Paragraph("Result", styles["CC_Head"]),
    ]]
    details = [
        ("Image filename", filename),
        ("Top predicted class", predicted_class),
        ("Top-class model score", f"{score * 100:.2f}%"),
        ("Orange gate confidence", f"{orange_probability * 100:.2f}%"),
        ("Gate acceptance threshold", f"{threshold * 100:.1f}%"),
        ("Report generated", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")),
    ]
    for label, value in details:
        rows.append([
            Paragraph(escape(str(label)), styles["CC_Body"]),
            Paragraph(escape(str(value)), styles["CC_Body"]),
        ])
    summary_table = Table(rows, colWidths=[5.1 * cm, 12.7 * cm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), forest),
        ("BACKGROUND", (0, 1), (-1, -1), ivory),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D8E4D5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([summary_table, Spacer(1, 5)])

    story.append(Paragraph("Treatment advice and next steps", styles["CC_Section"]))
    story.append(Paragraph(escape(advice["summary"]), styles["CC_Body"]))
    for action in advice["actions"]:
        story.append(Paragraph("&#8226;&nbsp; " + escape(action), styles["CC_Body"]))

    caution = (
        "Important: this is educational guidance based on a model prediction, "
        "not a confirmed plant diagnosis. Do not choose or apply pesticides "
        "from this report alone. Confirm the problem locally and follow current "
        "product labels, crop registrations, safety instructions and "
        "pre-harvest intervals."
    )
    caution_table = Table(
        [[Paragraph(escape(caution), styles["CC_Body"])]],
        colWidths=[17.8 * cm],
    )
    caution_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF2D9")),
        ("BOX", (0, 0), (-1, -1), .7, orange),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([Spacer(1, 4), caution_table])

    story.append(Paragraph("Scores for all classes", styles["CC_Section"]))
    score_rows = [[
        Paragraph("Disease class", styles["CC_Head"]),
        Paragraph("Model score", styles["CC_Head"]),
    ]]
    for label, value in sorted(class_scores, key=lambda item: item[1], reverse=True):
        score_rows.append([
            Paragraph(escape(str(label)), styles["CC_Body"]),
            Paragraph(f"{float(value) * 100:.2f}%", styles["CC_Body"]),
        ])
    scores_table = Table(score_rows, colWidths=[12 * cm, 5.8 * cm], repeatRows=1)
    scores_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), forest),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ivory]),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D8E4D5")),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(scores_table)
    story.append(Paragraph(
        "Scores are model outputs and are not necessarily calibrated "
        "probabilities or a verified likelihood of disease.",
        styles["CC_Small"],
    ))

    story.append(Paragraph("Image evidence", styles["CC_Section"]))
    image_buffers = []
    image_cells = []
    for array in (original, overlay):
        temp = BytesIO()
        Image.fromarray(array).save(temp, format="PNG")
        temp.seek(0)
        image_buffers.append(temp)
        image_cells.append(PDFImage(
            temp, width=8.1 * cm, height=5.7 * cm, kind="proportional"
        ))

    image_table = Table(
        [[
            Paragraph("Original image", styles["CC_Body"]),
            Paragraph("Grad-CAM overlay", styles["CC_Body"]),
        ], image_cells],
        colWidths=[8.9 * cm, 8.9 * cm],
    )
    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(image_table)
    story.append(Paragraph(
        "Grad-CAM is an explanation aid; it does not prove the location "
        "or cause of a plant disease.",
        styles["CC_Small"],
    ))

    story.append(Paragraph("Reference sources", styles["CC_Section"]))
    references = [
        ("ICAR citrus advisory",
         "https://icar.gov.in/sites/default/files/Circulars/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf"),
        ("ICAR-CCRI clean planting material",
         "https://icar.gov.in/index.php/en/icar-ccri-nagpur-organizes-national-workshop-clean-planting-material-citrus"),
        ("National Horticulture Board citrus bulletin",
         "https://nhb.gov.in/bulletin_files/fruits/citrus/cit002.pdf"),
    ]
    for title, url in references:
        story.append(Paragraph(
            f'<link href="{url}" color="#214B35"><u>{escape(title)}</u></link>',
            styles["CC_Small"],
        ))

    doc.build(story)
    return buffer.getvalue()


uploaded_file = st.file_uploader(
    "Drop an image here or browse files",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=440,
        )

        force_disease_prediction = st.checkbox(
            "I confirm this is an orange — continue if the gate rejects it",
            value=False,
            help=(
                "Use this when the image clearly shows an orange "
                "but the automatic detector gives low confidence."
            ),
        )

        if st.button("🍊  Analyze image", type="primary", use_container_width=True):
            with st.spinner("Checking the image and analyzing disease..."):
                model, grad_model, classes, gate_model, gate_config = (
                    load_resources()
                )

                resized = image.resize(IMAGE_SIZE)
                image_array = np.asarray(resized, dtype=np.float32)
                image_batch = image_array[np.newaxis, ...]

                # Step 1: check whether this image is likely to contain an orange.
                orange_probability = float(
                    gate_model.predict(image_batch, verbose=0)[0][0]
                )
                threshold = float(gate_config.get("threshold", 0.5))
                is_orange = orange_probability >= threshold

                # Allow prediction after automatic acceptance or explicit override.
                if is_orange or force_disease_prediction:
                    scores = model.predict(
                        image_batch,
                        verbose=0,
                    )[0]

                    predicted_index = int(np.argmax(scores))
                    predicted_class = classes[predicted_index]
                    score = float(scores[predicted_index])

                    heatmap = create_gradcam(
                        grad_model,
                        image_batch,
                        predicted_index,
                    )

                    heatmap_rgb = (
                        colormaps["jet"](heatmap)[..., :3] * 255
                    ).astype(np.uint8)

                    original = np.asarray(resized, dtype=np.uint8)
                    overlay = (
                        0.60 * original + 0.40 * heatmap_rgb
                    ).clip(0, 255).astype(np.uint8)

            st.subheader("Input validation")
            st.metric(
                "Orange confidence",
                f"{orange_probability * 100:.2f}%",
            )
            st.caption(
                f"Orange gate threshold: {threshold * 100:.1f}%"
            )

            if not is_orange and not force_disease_prediction:
                st.warning(
                    "This image was not accepted as an orange. "
                    "Disease classification was skipped."
                )
                st.info(
                    "Try a clear image showing an orange fruit. "
                    "Images outside the detector's training data "
                    "may still be misclassified."
                )
            else:
                if not is_orange:
                    st.warning(
                        "Manual override used. The gate had low confidence; "
                        "please treat the disease prediction cautiously."
                    )
                else:
                    st.success("Orange image accepted.")

                st.subheader("Disease prediction")

                if score < 0.90:
                    st.warning(
                        f"Uncertain prediction: {predicted_class}. "
                        "The top-class model score is below 90%. "
                        "Please verify the result."
                    )
                else:
                    st.success(f"Predicted class: {predicted_class}")

                st.metric(
                    "Top-class model score",
                    f"{score * 100:.2f}%",
                )

                st.info(
                    "This score is not a calibrated probability. "
                    "Even high-scoring predictions can be incorrect, "
                    "particularly for diseases with similar symptoms."
                )

                st.markdown(
                    '<div class="section-kicker">STAGE 03 · EXPLANATION</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-title">Explore the analysis</div>',
                    unsafe_allow_html=True,
                )

                original_tab, heatmap_tab, scores_tab = st.tabs([
                    "🖼️  Original image",
                    "✨  Grad-CAM",
                    "📊  Class scores",
                ])

                with original_tab:
                    st.image(
                        original,
                        caption="Resized input image",
                        width=520,
                    )
                    st.caption(
                        "This is the image supplied to the disease classifier "
                        "after resizing to 224 × 224 pixels."
                    )

                with heatmap_tab:
                    st.image(
                        overlay,
                        caption="Grad-CAM overlay",
                        width=520,
                    )
                    st.caption(
                        "Warm regions indicate stronger positive contributions "
                        "to the selected class. This visualization is an aid, "
                        "not proof of disease location."
                    )

                with scores_tab:
                    st.caption(
                        "Compare all disease categories. These model scores "
                        "are not necessarily calibrated probabilities."
                    )
                    for index in np.argsort(scores)[::-1]:
                        label = str(classes[index]).replace("_", " ").title()
                        value = float(scores[index])
                        score_col, value_col = st.columns([4, 1])
                        with score_col:
                            st.markdown(f"**{label}**")
                            bar_class = "score-fill-top" if index == predicted_index else "score-fill-other"
                            st.markdown(
                            f'<div class="score-track" role="img" '
                            f'aria-label="{label}: {value * 100:.1f}%">'
                            f'<div class="{bar_class}" '
                            f'style="width:{max(0.0, min(100.0, value * 100.0)):.2f}%;">'
                            '</div></div>',
                            unsafe_allow_html=True,
                        )
                        with value_col:
                            st.markdown(
                                f"<div style='text-align:right;padding-top:.2rem;"
                                f"font-weight:800;color:#214B35;'>"
                                f"{value * 100:.1f}%</div>",
                                unsafe_allow_html=True,
                            )
                    st.caption(
                        "Unfamiliar images or diseases outside the training "
                        "data may produce unreliable results."
                    )


                st.markdown("### Treatment advice and next steps")
                advice = get_treatment_advice(predicted_class)
                st.info(advice["summary"])
                for advice_item in advice["actions"]:
                    st.markdown(f"- {advice_item}")

                st.warning(
                    "Educational guidance only. Confirm the disease with "
                    "a local horticulture expert or KVK before treatment. "
                    "Do not apply pesticides based only on this AI result."
                )

                st.caption("Reference guidance: ICAR, ICAR-CCRI and the National Horticulture Board.")
                st.markdown(
                    "[ICAR citrus advisory](https://icar.gov.in/sites/default/files/Circulars/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf)"
                    " · [ICAR-CCRI](https://icar.gov.in/index.php/en/icar-ccri-nagpur-organizes-national-workshop-clean-planting-material-citrus)"
                    " · [NHB citrus bulletin](https://nhb.gov.in/bulletin_files/fruits/citrus/cit002.pdf)"
                )

                try:
                    report_scores = [
                        (str(name).replace("_", " ").title(), float(value))
                        for name, value in zip(classes, scores)
                    ]
                    report_bytes = create_treatment_pdf(
                        filename=uploaded_file.name,
                        predicted_class=str(predicted_class).replace("_", " ").title(),
                        score=score,
                        orange_probability=orange_probability,
                        threshold=threshold,
                        class_scores=report_scores,
                        advice=advice,
                        original=original,
                        overlay=overlay,
                    )
                    st.download_button(
                        "📄 Download treatment and prediction report (PDF)",
                        data=report_bytes,
                        file_name="citruscare_disease_report.pdf",
                        mime="application/pdf",
                        key="citruscare_pdf_report",
                        type="primary",
                        use_container_width=True,
                    )
                except ModuleNotFoundError:
                    st.warning(
                        "PDF export requires ReportLab. Install it with "
                        "`pip install reportlab` and restart the app."
                    )
                except Exception as report_error:
                    st.error(f"Could not create the PDF report: {report_error}")


    except Exception as error:
        st.error(f"Unable to process this image: {error}")
        st.exception(error)
