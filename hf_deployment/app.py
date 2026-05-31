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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.config import (
    CV_CONFIG, APP_CONFIG, CLASS_NAMES, HAM10000_CLASSES,
    ANTHROPIC_API_KEY,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data/raw/HAM10000_images_part_1"

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
            ml_path = str(f); break
        _pipeline = SkinLesionPipeline(
            cv_model_path=cv_path,
            ml_model_path=ml_path,
            device=os.getenv("DEVICE", "cpu"),
            use_llm=bool(ANTHROPIC_API_KEY),
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
        return None, "⬆️ Bitte ein Bild hochladen.", "", "", ""

    try:
        pipeline = get_pipeline()
        metadata = {
            "age": age, "sex": sex.lower(),
            "localization": localization, "localization_enc": -1,
        }
        result = pipeline.predict(
            image=image, symptom_text=symptom_text,
            metadata=metadata,
            generate_explanation=bool(symptom_text and ANTHROPIC_API_KEY),
        )

        cv       = result.get("cv", {})
        ml       = result.get("ml", {})
        final    = result.get("final_label", cv.get("label", "unknown"))
        final_nm = HAM10000_CLASSES.get(final, final)
        conf     = cv.get("confidence", 0.0)
        is_risk  = final in HIGH_RISK

        risk_icon = "🔴" if is_risk else "🟢"
        risk_label = "HOHES RISIKO" if is_risk else "GERINGES RISIKO"

        summary = (
            f"## {risk_icon} {final_nm}\n"
            f"**Klasse:** `{final}` &nbsp;|&nbsp; "
            f"**Konfidenz:** {conf:.1%} &nbsp;|&nbsp; "
            f"**Risiko:** {risk_label}\n\n"
        )
        if ml:
            ml_nm = HAM10000_CLASSES.get(ml.get('label', final), final_nm)
            summary += f"**ML Ensemble:** {ml_nm} &nbsp;|&nbsp; Risk Score: `{ml.get('risk_score', 0):.2f}`\n"

        top3_md = "### Top-3 Diagnosen\n| Klasse | Name | Wahrscheinlichkeit |\n|---|---|---|\n"
        for cls, name, prob in cv.get("top_k", [])[:3]:
            icon = "🔴" if cls in HIGH_RISK else "🟢"
            top3_md += f"| `{cls}` | {icon} {name} | **{prob:.1%}** |\n"

        nlp = result.get("nlp", {})
        nlp_md = ""
        if nlp and "duration_days" in nlp:
            nlp_md = "### Extrahierte Symptom-Features\n"
            for k, v in nlp.items():
                if v not in (-1, "unknown", "", None):
                    nlp_md += f"- **{k}:** {v}\n"

        explanation = result.get("explanation", "")

        chart = make_probability_chart(cv.get("probabilities", {}))
        return chart, summary, top3_md, explanation, nlp_md

    except Exception as e:
        import traceback
        return None, f"**Fehler:** {str(e)}\n```\n{traceback.format_exc()}\n```", "", "", ""


# ── Example images ────────────────────────────────────────────────────────────
EXAMPLES = [
    [str(RAW_DIR / "ISIC_0026188.jpg"), "dark irregular spot growing for months", 70, "Male",   "trunk"],
    [str(RAW_DIR / "ISIC_0027291.jpg"), "pearly nodule on back, bleeds sometimes", 70, "Male",   "back"],
    [str(RAW_DIR / "ISIC_0025803.jpg"), "rough scaly red patch on face",           80, "Female", "face"],
    [str(RAW_DIR / "ISIC_0026189.jpg"), "stable brown mole, unchanged for years",  65, "Male",   "back"],
]

# ── Layout ────────────────────────────────────────────────────────────────────
CSS = """
.risk-high { color: #e74c3c; font-weight: bold; }
.disclaimer { background: #2c0000; border-left: 4px solid #e74c3c;
              padding: 10px 16px; border-radius: 4px; color: #ffcccc; }
footer { display: none !important; }
"""

with gr.Blocks(title="Skin Lesion Classifier", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("""
# 🔬 Skin Lesion Classifier
### ZHAW AI-Applications – ResNet50 + ML Ensemble + NLP
""")

    gr.HTML(f"""
<div class="disclaimer">
⚠️ <strong>DISCLAIMER:</strong> {APP_CONFIG['disclaimer']}
</div>
""")

    with gr.Row():
        # ── Left: Input ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Input")
            image_input = gr.Image(
                type="pil", label="Hautläsion Bild hochladen",
                elem_id="img-upload",
            )
            symptom_input = gr.Textbox(
                label="Symptombeschreibung (optional)",
                placeholder="z.B. dunkler unregelmässiger Fleck, wächst seit 3 Monaten, manchmal juckend...",
                lines=3,
                info="Freitext – verbessert die ML-Ensemble-Vorhersage",
            )
            with gr.Row():
                age_input = gr.Slider(0, 100, value=45, step=1, label="Alter",
                                      info="Patientenalter in Jahren")
                sex_input = gr.Radio(["Male", "Female"], value="Male", label="Geschlecht")
            loc_input = gr.Dropdown(
                choices=["face","scalp","neck","chest","back","trunk","abdomen",
                         "upper extremity","lower extremity","hand","foot","unknown"],
                value="unknown", label="Lokalisation",
                info="Körperstelle der Läsion",
            )
            submit_btn = gr.Button("🔍 Analysieren", variant="primary", size="lg")

            gr.Markdown("#### 📋 Beispiele")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[image_input, symptom_input, age_input, sex_input, loc_input],
                label="Beispiel-Bilder aus HAM10000",
            )

        # ── Right: Output ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Ergebnisse")
            chart_out   = gr.Plot(label="Klassenwahrscheinlichkeiten")
            summary_out = gr.Markdown(label="Vorhersage")
            top3_out    = gr.Markdown(label="Top-3")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("🤖 KI-Erklärung", open=True):
                expl_out = gr.Textbox(label="", lines=5, show_label=False)
        with gr.Column(scale=1):
            with gr.Accordion("🧬 NLP Symptom-Features", open=False):
                nlp_out = gr.Markdown()

    submit_btn.click(
        fn=predict,
        inputs=[image_input, symptom_input, age_input, sex_input, loc_input],
        outputs=[chart_out, summary_out, top3_out, expl_out, nlp_out],
    )

    gr.Markdown("---\n*ZHAW School of Engineering – AI-Applications FS2025*")


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=APP_CONFIG["port"],
        share=False,
    )
