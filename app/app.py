"""
app/app.py - Gradio web interface for Skin Lesion Classifier
ZHAW AI-Applications Project
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
load_dotenv()

from src.config import (
    CV_CONFIG, APP_CONFIG, CLASS_NAMES, HAM10000_CLASSES,
    ANTHROPIC_API_KEY, OPENAI_API_KEY,
)

HIGH_RISK = {"mel", "bcc", "akiec"}
RISK_COLOR = {
    "mel":   "#e74c3c",
    "bcc":   "#e67e22",
    "akiec": "#f39c12",
    "bkl":   "#3498db",
    "nv":    "#27ae60",
    "df":    "#27ae60",
    "vasc":  "#9b59b6",
}

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline import SkinLesionPipeline
        cv_path = os.getenv("CV_MODEL_PATH", CV_CONFIG["best_model"])
        ml_path = None
        for f in sorted(Path("models").glob("ml_*.pkl")):
            ml_path = str(f)
            break
        _pipeline = SkinLesionPipeline(
            cv_model_path=cv_path,
            ml_model_path=ml_path,
            device=os.getenv("DEVICE", "cpu"),
            use_llm=bool(OPENAI_API_KEY),
        )
    return _pipeline


def make_probability_chart(probabilities: dict) -> plt.Figure:
    labels = list(HAM10000_CLASSES.values())
    probs  = [probabilities.get(k, 0.0) for k in CLASS_NAMES]
    colors = [RISK_COLOR.get(k, "#95a5a6") for k in CLASS_NAMES]
    sorted_pairs = sorted(zip(probs, labels, colors), reverse=True)
    probs_s, labels_s, colors_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.barh(labels_s, probs_s, color=colors_s, edgecolor="none", height=0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, prob in zip(bars, probs_s):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_title("Class Probabilities (ResNet50)", color="white", fontsize=11, pad=8)
    plt.tight_layout()
    return fig


def predict(image, symptom_text, age, sex, localization):
    if image is None:
        return None, "⬆️ Please upload an image.", "", "", ""

    try:
        pipeline = get_pipeline()
        metadata = {
            "age": age,
            "sex": sex.lower(),
            "localization": localization,
            "localization_enc": -1,
        }
        result = pipeline.predict(
            image=image,
            symptom_text=symptom_text,
            metadata=metadata,
            generate_explanation=bool(symptom_text and OPENAI_API_KEY),
        )

        cv      = result.get("cv", {})
        ml      = result.get("ml", {})
        final   = result.get("final_label", cv.get("label", "unknown"))
        final_nm = HAM10000_CLASSES.get(final, final)
        conf    = cv.get("confidence", 0.0)
        is_risk = final in HIGH_RISK

        risk_icon  = "🔴" if is_risk else "🟢"
        risk_label = "HIGH RISK" if is_risk else "LOW RISK"

        summary = (
            f"## {risk_icon} {final_nm}\n"
            f"**Class:** `{final}` &nbsp;|&nbsp; "
            f"**Confidence:** {conf:.1%} &nbsp;|&nbsp; "
            f"**Risk:** {risk_label}\n\n"
        )
        if ml:
            ml_nm = HAM10000_CLASSES.get(ml.get("label", final), final_nm)
            summary += f"**ML Ensemble:** {ml_nm} &nbsp;|&nbsp; Risk Score: `{ml.get('risk_score', 0):.2f}`\n"

        top3_md = "### Top-3 Diagnoses\n| Class | Name | Probability |\n|---|---|---|\n"
        for cls, name, prob in cv.get("top_k", [])[:3]:
            icon = "🔴" if cls in HIGH_RISK else "🟢"
            top3_md += f"| `{cls}` | {icon} {name} | **{prob:.1%}** |\n"

        # NLP features — skip raw similarity scores
        nlp = result.get("nlp", {})
        nlp_md = ""
        if nlp:
            useful = {
                k: v for k, v in nlp.items()
                if k != "class_similarities"
                and v not in (-1, "unknown", "", None, False, 0)
            }
            if useful:
                nlp_md = "### Extracted Symptom Features\n"
                for k, v in useful.items():
                    nlp_md += f"- **{k.replace('_', ' ').title()}:** {v}\n"
            else:
                nlp_md = "### Extracted Symptom Features\n*Text analysed — no specific features extracted.*"

        explanation = result.get("explanation", "")

        chart = make_probability_chart(cv.get("probabilities", {}))
        return chart, summary, top3_md, explanation, nlp_md

    except Exception as e:
        import traceback
        return None, f"**Error:** {str(e)}\n```\n{traceback.format_exc()}\n```", "", "", ""


# ── Layout ────────────────────────────────────────────────────────────────────
CSS = """
.disclaimer {
    background: #2c0000;
    border-left: 4px solid #e74c3c;
    padding: 10px 16px;
    border-radius: 4px;
    color: #ffcccc;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Skin Lesion Classifier", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("""
# 🔬 Skin Lesion Classifier
### ZHAW AI-Applications — ResNet50 + ML Ensemble + NLP
""")

    gr.HTML(f"""
<div class="disclaimer">
⚠️ <strong>DISCLAIMER:</strong> {APP_CONFIG['disclaimer']}
</div>
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            image_input = gr.Image(type="pil", label="Upload skin lesion image")
            symptom_input = gr.Textbox(
                label="Symptom description (optional)",
                placeholder="e.g. dark irregular spot growing for 6 months, occasionally itchy...",
                lines=3,
                info="Free text — improves the ML ensemble prediction",
            )
            with gr.Row():
                age_input = gr.Slider(0, 100, value=45, step=1, label="Age")
                sex_input = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            loc_input = gr.Dropdown(
                choices=["face", "scalp", "neck", "chest", "back", "trunk",
                         "abdomen", "upper extremity", "lower extremity",
                         "hand", "foot", "unknown"],
                value="unknown",
                label="Localization",
                info="Body site of the lesion",
            )
            submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            chart_out   = gr.Plot(label="Class Probabilities")
            summary_out = gr.Markdown(label="Prediction")
            top3_out    = gr.Markdown(label="Top-3")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("🤖 AI Explanation", open=True):
                expl_out = gr.Textbox(label="", lines=5, show_label=False)
        with gr.Column(scale=1):
            with gr.Accordion("🧬 NLP Symptom Features", open=True):
                nlp_out = gr.Markdown()

    submit_btn.click(
        fn=predict,
        inputs=[image_input, symptom_input, age_input, sex_input, loc_input],
        outputs=[chart_out, summary_out, top3_out, expl_out, nlp_out],
    )

    gr.Markdown("---\n*ZHAW School of Engineering — AI-Applications FS2025*")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)