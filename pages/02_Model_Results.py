from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config.yaml"
HISTORY_PATH = REPORTS / "finetune_history.csv"
TEST_METRICS_PATH = REPORTS / "test_metrics.txt"
CLASS_REPORT_PATH = REPORTS / "classification_report.csv"
CONFUSION_PATH = REPORTS / "confusion_matrix.png"
GATE_METRICS_PATH = REPORTS / "orange_gate_test_metrics.json"

st.set_page_config(
    page_title="Model Results | CitrusCare",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #FBF8F1; color: #26382D; }
    h1, h2, h3 { color: #214B35; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E1E8DE;
        border-radius: 14px; padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #526858 !important; }
    [data-testid="stMetricValue"] { color: #214B35 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Training & Model Results")
st.caption(
    "CitrusCare · Metrics and plots loaded from this project's saved training history and evaluation reports."
)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def count_images(split_path: Path) -> int | None:
    if not split_path.is_dir():
        return None
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(1 for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in valid)


def parse_test_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for label, key in [("Test loss", "loss"), ("Test accuracy", "accuracy")]:
        match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)", text, re.IGNORECASE)
        if match:
            values[key] = float(match.group(1))
    return values


cfg = load_yaml(CONFIG_PATH)
history = None
if HISTORY_PATH.exists():
    try:
        history = pd.read_csv(HISTORY_PATH)
    except Exception as exc:
        st.warning(f"Could not read the training history CSV: {exc}")

metrics = parse_test_metrics(TEST_METRICS_PATH)
class_report = None
if CLASS_REPORT_PATH.exists():
    try:
        class_report = pd.read_csv(CLASS_REPORT_PATH, index_col=0)
    except Exception as exc:
        st.warning(f"Could not read the class-wise report: {exc}")

# Exact batch sizes are read from this project's config and training code.
configured_batch = int(cfg.get("batch_size", 32))
finetune_batch = min(configured_batch, 16)  # src/finetune.py uses this cap.
gate_batch = 16  # src/train_orange_gate.py defines BATCH_SIZE = 16.
image_size = cfg.get("image_size", [224, 224])
initial_epochs = int(cfg.get("initial_epochs", 15))
finetune_epochs = int(cfg.get("fine_tune_epochs", 15))

st.subheader("Training configuration")
settings = st.columns(4)
settings[0].metric("Input size", f"{image_size[0]} × {image_size[1]}")
settings[1].metric("Baseline batch size", str(configured_batch))
settings[2].metric("Fine-tuning batch size", str(finetune_batch))
settings[3].metric("Orange gate batch size", str(gate_batch))
st.caption(
    f"Configured baseline epochs: {initial_epochs} · Configured fine-tuning epochs: {finetune_epochs}. "
    "The saved disease-model history below contains the fine-tuning run, not a separate epoch history for baseline training."
)

# Show local data split counts if the data folders are present.
data_root = ROOT / "data" / "processed"
split_counts = {split: count_images(data_root / split) for split in ("train", "val", "test")}
if any(value is not None for value in split_counts.values()):
    st.subheader("Dataset split")
    cols = st.columns(3)
    labels = [("Training images", "train"), ("Validation images", "val"), ("Test images", "test")]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, f"{split_counts[key]:,}" if split_counts[key] is not None else "Not found")
else:
    st.info("The data/processed folders were not found here, so image counts cannot be calculated on this machine.")

st.divider()
st.subheader("Fine-tuning learning curves")
if history is not None and not history.empty:
    required = {"accuracy", "val_accuracy", "loss", "val_loss"}
    missing = required.difference(history.columns)
    if missing:
        st.warning(f"Training-history file is missing columns: {', '.join(sorted(missing))}")
    else:
        plot_history = history.copy()
        if "epoch" in plot_history.columns:
            plot_history["epoch_display"] = plot_history["epoch"].astype(int) + 1
        else:
            plot_history["epoch_display"] = range(1, len(plot_history) + 1)

        fig_acc, ax_acc = plt.subplots(figsize=(8.5, 4.2))
        ax_acc.plot(plot_history["epoch_display"], plot_history["accuracy"] * 100,
                    marker="o", linewidth=2.2, color="#214B35", label="Training accuracy")
        ax_acc.plot(plot_history["epoch_display"], plot_history["val_accuracy"] * 100,
                    marker="o", linewidth=2.2, color="#F28C28", label="Validation accuracy")
        ax_acc.set_title("Accuracy by fine-tuning epoch")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 100)
        ax_acc.grid(True, alpha=.22)
        ax_acc.legend(frameon=False)
        fig_acc.tight_layout()
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

        fig_loss, ax_loss = plt.subplots(figsize=(8.5, 4.2))
        ax_loss.plot(plot_history["epoch_display"], plot_history["loss"],
                     marker="o", linewidth=2.2, color="#214B35", label="Training loss")
        ax_loss.plot(plot_history["epoch_display"], plot_history["val_loss"],
                     marker="o", linewidth=2.2, color="#F28C28", label="Validation loss")
        ax_loss.set_title("Loss by fine-tuning epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=.22)
        ax_loss.legend(frameon=False)
        fig_loss.tight_layout()
        st.pyplot(fig_loss, use_container_width=True)
        plt.close(fig_loss)

        best_val_acc_idx = plot_history["val_accuracy"].idxmax()
        best_val_loss_idx = plot_history["val_loss"].idxmin()
        best_val_acc_row = plot_history.loc[best_val_acc_idx]
        best_val_loss_row = plot_history.loc[best_val_loss_idx]
        summary_cols = st.columns(3)
        summary_cols[0].metric(
            "Best training accuracy",
            f"{plot_history['accuracy'].max() * 100:.2f}%",
        )
        summary_cols[1].metric(
            "Best validation accuracy",
            f"{best_val_acc_row['val_accuracy'] * 100:.2f}%",
            help=f"Reached at epoch {int(best_val_acc_row['epoch_display'])}.",
        )
        summary_cols[2].metric(
            "Lowest validation loss",
            f"{best_val_loss_row['val_loss']:.4f}",
            help=f"Reached at epoch {int(best_val_loss_row['epoch_display'])}.",
        )
        st.download_button(
            "Download fine-tuning history CSV",
            data=HISTORY_PATH.read_bytes(),
            file_name="finetune_history.csv",
            mime="text/csv",
        )
else:
    st.warning(
        "reports/finetune_history.csv was not found. The fine-tuning curves require this saved training history; "
        "the page will not fabricate an epoch curve."
    )
    saved_curve = REPORTS / "finetune_curves.png"
    if saved_curve.exists():
        st.image(saved_curve, caption="Saved fine-tuning curves")

st.divider()
st.subheader("Final test-set evaluation")
if metrics:
    m1, m2, m3 = st.columns(3)
    if "accuracy" in metrics:
        m1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%", help="95 correct predictions out of 99 test images.")
    if "loss" in metrics:
        m2.metric("Test loss", f"{metrics['loss']:.6f}")
    if class_report is not None and "macro avg" in class_report.index and "f1-score" in class_report.columns:
        m3.metric("Macro F1-score", f"{class_report.loc['macro avg', 'f1-score'] * 100:.2f}%")
    st.caption(
        "The test set is held out for final evaluation, not monitored every epoch. "
        "For that reason, this page shows the final test metrics and confusion matrix rather than a test curve."
    )
else:
    st.warning("reports/test_metrics.txt was not found, so final test metrics could not be loaded.")

if class_report is not None:
    wanted_classes = cfg.get("class_names", ["Black_Spot", "Canker", "Greening", "Healthy", "Melanose"])
    present = [name for name in wanted_classes if name in class_report.index]
    if present:
        st.markdown("#### Per-class performance")
        metric_columns = [name for name in ["precision", "recall", "f1-score", "support"] if name in class_report.columns]
        display_df = class_report.loc[present, metric_columns].copy()
        for col in ["precision", "recall", "f1-score"]:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(2)
        rename = {"precision": "Precision (%)", "recall": "Recall (%)", "f1-score": "F1 (%)", "support": "Test images"}
        st.dataframe(display_df.rename(columns=rename), use_container_width=True)

        if all(col in class_report.columns for col in ["precision", "recall", "f1-score"]):
            chart_df = class_report.loc[present, ["precision", "recall", "f1-score"]] * 100
            chart_df = chart_df.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"})
            st.bar_chart(chart_df, use_container_width=True)

confusion_path = REPORTS / "confusion_matrix.png"
if confusion_path.exists():
    st.markdown("#### Confusion matrix")
    st.image(confusion_path, caption="Final test-set confusion matrix", use_container_width=True)

# Optional results from the separate binary orange gate.
if GATE_METRICS_PATH.exists():
    try:
        gate = json.loads(GATE_METRICS_PATH.read_text(encoding="utf-8"))
        chosen = gate.get("test_metrics_at_validation_selected_threshold", {})
        if chosen:
            st.divider()
            st.subheader("Orange input-gate evaluation")
            gc = st.columns(4)
            gc[0].metric("Selected threshold", f"{float(chosen.get('threshold', 0)) * 100:.2f}%")
            gc[1].metric("Gate test accuracy", f"{float(chosen.get('accuracy', 0)) * 100:.2f}%")
            gc[2].metric("Precision", f"{float(chosen.get('precision', 0)) * 100:.2f}%")
            gc[3].metric("Recall", f"{float(chosen.get('recall', 0)) * 100:.2f}%")
            st.warning(
                "These are controlled test-set results. Separate real-world checks found false acceptance and rejection cases, "
                "so the gate is not yet reliable enough to treat as a definitive orange detector."
            )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        st.warning(f"Orange gate metrics could not be read: {exc}")

# Both stages of the separate binary orange-gate model have saved CSV histories.
with st.expander("Orange input-gate training curves", expanded=False):
    gate_history_paths = [
        ("Stage 1 · classifier head", REPORTS / "orange_gate_stage1_history.csv"),
        ("Stage 2 · fine-tuning", REPORTS / "orange_gate_stage2_history.csv"),
    ]
    found_gate_history = False
    for stage_title, stage_path in gate_history_paths:
        if not stage_path.exists():
            continue
        try:
            gate_history = pd.read_csv(stage_path)
        except Exception as exc:
            st.warning(f"Could not read {stage_path.name}: {exc}")
            continue
        if gate_history.empty or not {"accuracy", "val_accuracy", "loss", "val_loss"}.issubset(gate_history.columns):
            st.warning(f"{stage_path.name} does not contain the expected curve columns.")
            continue
        found_gate_history = True
        gate_history["epoch_display"] = gate_history["epoch"].astype(int) + 1 if "epoch" in gate_history else range(1, len(gate_history) + 1)
        st.markdown(f"#### {stage_title}")
        gate_acc_fig, gate_acc_ax = plt.subplots(figsize=(8.5, 3.5))
        gate_acc_ax.plot(gate_history["epoch_display"], gate_history["accuracy"] * 100,
                         color="#214B35", marker="o", label="Training accuracy")
        gate_acc_ax.plot(gate_history["epoch_display"], gate_history["val_accuracy"] * 100,
                         color="#F28C28", marker="o", label="Validation accuracy")
        gate_acc_ax.set_xlabel("Epoch")
        gate_acc_ax.set_ylabel("Accuracy (%)")
        gate_acc_ax.set_ylim(0, 100)
        gate_acc_ax.grid(True, alpha=.22)
        gate_acc_ax.legend(frameon=False)
        gate_acc_fig.tight_layout()
        st.pyplot(gate_acc_fig, use_container_width=True)
        plt.close(gate_acc_fig)

        gate_loss_fig, gate_loss_ax = plt.subplots(figsize=(8.5, 3.5))
        gate_loss_ax.plot(gate_history["epoch_display"], gate_history["loss"],
                          color="#214B35", marker="o", label="Training loss")
        gate_loss_ax.plot(gate_history["epoch_display"], gate_history["val_loss"],
                          color="#F28C28", marker="o", label="Validation loss")
        gate_loss_ax.set_xlabel("Epoch")
        gate_loss_ax.set_ylabel("Loss")
        gate_loss_ax.grid(True, alpha=.22)
        gate_loss_ax.legend(frameon=False)
        gate_loss_fig.tight_layout()
        st.pyplot(gate_loss_fig, use_container_width=True)
        plt.close(gate_loss_fig)
    if not found_gate_history:
        st.info("No orange-gate stage history CSVs were found under reports/.")

st.divider()
st.caption(
    "CitrusCare is an educational prototype. Its class scores are model outputs, not calibrated probabilities, "
    "and the application is not a substitute for expert agricultural diagnosis."
)
