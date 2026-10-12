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


st.markdown("""
<style>
/* ORANGE_RESPONSIVE_STYLE */

.block-container {
    max-width: 1120px !important;
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

.stApp {
    font-size: 16px;
}

h1 {
    font-size: 2.25rem !important;
    line-height: 1.25 !important;
}

h2 {
    font-size: 1.65rem !important;
}

h3 {
    font-size: 1.3rem !important;
}

.stMarkdown p,
.stCaption,
[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

[data-testid="stImage"] img {
    max-height: 420px !important;
    max-width: 100% !important;
    object-fit: contain !important;
}

.stButton button {
    min-height: 2.7rem;
    font-size: 1rem !important;
    padding: 0.55rem 1.1rem !important;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    h1 {
        font-size: 1.8rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("🍊 Orange Fruit Disease Detector")
st.write(
    "Upload an orange fruit image to classify it using "
    "our fine-tuned EfficientNet-B0 model."
)


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


uploaded_file = st.file_uploader(
    "Upload an orange image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width="stretch",
        )

        force_disease_prediction = st.checkbox(
            "I confirm this is an orange — continue if the gate rejects it",
            value=False,
            help=(
                "Use this when the image clearly shows an orange "
                "but the automatic detector gives low confidence."
            ),
        )

        if st.button("Predict disease", type="primary"):
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

                st.subheader("Grad-CAM explanation")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        original,
                        caption="Resized input image",
                        width="stretch",
                    )

                with col2:
                    st.image(
                        overlay,
                        caption="Grad-CAM overlay",
                        width="stretch",
                    )

                st.caption(
                    "Warm regions generally indicate stronger positive "
                    "contributions to the selected class. The heatmap "
                    "is an explanation aid, not proof of disease "
                    "location or a confirmed agricultural diagnosis."
                )

                st.subheader("Scores for all disease classes")
                for index in np.argsort(scores)[::-1]:
                    st.write(
                        f"**{classes[index]}** — "
                        f"{scores[index] * 100:.2f}%"
                    )
                    st.progress(float(scores[index]))

                st.caption(
                    "Disease model scores may not be calibrated "
                    "probabilities. Results may be unreliable for "
                    "unseen diseases or unfamiliar images."
                )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")
        st.exception(error)
