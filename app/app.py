"""
app/app.py - Gradio web interface for Skin Lesion Classifier
ZHAW AI-Applications Project

Usage:
    python app/app.py
    # Then open: http://localhost:7860
"""
import sys
import os
from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.config import (
    CV_CONFIG, APP_CONFIG, CLASS_NAMES, HAM10000_CLASSES,
    ANTHROPIC_API_KEY
)

# ── Global pipeline (loaded once) ──────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline import SkinLesionPipeline
        cv_path = os.getenv("CV_MODEL_PATH", CV_CONFIG["best_model"])
        ml_path = None
        for f in Path("models").glob("ml_*.pkl"):
            ml_path = str(f)
            break
        _pipeline = SkinLesionPipeline(
            cv_model_path=cv_path,
            ml_model_path=ml_path,
            device=os.getenv("DEVICE", "cpu"),
            use_llm=bool(ANTHROPIC_API_KEY),
        )
    return _pipeline


def make_probability_chart(probabilities: dict) -> plt.Figure:
    """Create a horizontal bar chart of class probabilities."""
    classes = list(HAM10000_CLASSES.values())
    probs   = [probabilities.get(k, 0.0) for k in CLASS_NAMES]
    colors  = ["#e74c3c" if p == max(probs) else "#3498db" for p in probs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("ResNet50 Class Probabilities")
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def predict(
    image: Image.Image,
    symptom_text: str,
    age: float,
    sex: str,
    localization: str,
) -> tuple:
    """Main prediction function called by Gradio."""
    if image is None:
        return None, "Please upload an image.", "", "", None

    try:
        pipeline = get_pipeline()
        metadata = {
            "age":             age,
            "sex":             sex.lower(),
            "localization":    localization,
            "localization_enc": -1,
        }

        result = pipeline.predict(
            image=image,
            symptom_text=symptom_text,
            metadata=metadata,
            generate_explanation=bool(symptom_text and ANTHROPIC_API_KEY),
        )

        cv       = result.get("cv", {})
        ml       = result.get("ml", {})
        final    = result.get("final_label", cv.get("label", "unknown"))
        final_nm = HAM10000_CLASSES.get(final, final)
        conf     = result.get("risk_score", cv.get("confidence", 0.0))

        # Summary
        summary = f"**Prediction: {final_nm} ({final})**\n"
        summary += f"Confidence: {conf:.1%}\n"
        if ml:
            summary += f"Risk Score: {ml.get("risk_score", conf):.2f}\n"

        # Top-3
        top3_text = "**Top-3 CV Predictions:**\n"
        for cls, name, prob in cv.get("top_k", [])[:3]:
            top3_text += f"- {name} ({cls}): {prob:.1%}\n"

        # NLP features
        nlp = result.get("nlp", {})
        nlp_text = ""
        if nlp and isinstance(nlp, dict) and "duration_days" in nlp:
            nlp_text = "**Extracted Symptoms:**\n"
            for k, v in nlp.items():
                if v not in (-1, "unknown", "") and v is not None:
                    nlp_text += f"- {k}: {v}\n"

        # Explanation
        explanation = result.get("explanation", APP_CONFIG["disclaimer"])

        # Probability chart
        chart = make_probability_chart(cv.get("probabilities", {}))

        return chart, summary, top3_text, explanation, nlp_text

    except Exception as e:
        err = f"Error: {str(e)}\n\n{APP_CONFIG["disclaimer"]}"
        return None, err, "", "", ""


# ── Gradio Interface ────────────────────────────────────────────────────────
DISCLAIMER_MD = f"""
> {APP_CONFIG["disclaimer"]}
"""

with gr.Blocks(
    title="Skin Lesion Classifier - ZHAW AI-Applications",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# Skin Lesion Classifier")
    gr.Markdown("**ZHAW AI-Applications Abschlussprojekt**")
    gr.Markdown(DISCLAIMER_MD)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Skin Lesion Image")
            symptom_text = gr.Textbox(
                label="Describe Your Symptoms (optional)",
                placeholder="e.g. Dark irregular mole on my back, growing for 3 months, slightly itchy...",
                lines=3,
            )
            with gr.Row():
                age_input = gr.Slider(minimum=0, maximum=100, value=45, label="Age")
                sex_input = gr.Radio(choices=["Male", "Female"], value="Male", label="Sex")
            loc_input = gr.Dropdown(
                choices=["face", "scalp", "neck", "chest", "back", "abdomen",
                         "upper extremity", "lower extremity", "hand", "foot", "unknown"],
                value="unknown",
                label="Localization",
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            chart_out      = gr.Plot(label="Class Probabilities")
            summary_out    = gr.Markdown(label="Prediction Summary")
            top3_out       = gr.Markdown(label="Top-3 Predictions")

    with gr.Row():
        nlp_out = gr.Markdown(label="Extracted Symptom Features")
        expl_out = gr.Textbox(label="AI Explanation", lines=6)

    submit_btn.click(
        fn=predict,
        inputs=[image_input, symptom_text, age_input, sex_input, loc_input],
        outputs=[chart_out, summary_out, top3_out, expl_out, nlp_out],
    )

    gr.Markdown("---")
    gr.Markdown(f"*{APP_CONFIG["disclaimer"]}*")


if __name__ == "__main__":
    demo.launch(
        server_name=APP_CONFIG["host"],
        server_port=APP_CONFIG["port"],
        share=APP_CONFIG["share"],
    )
