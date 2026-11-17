
import io, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
HISTORY_FILE = REPORTS / "prediction_history.json"
CLASS_FILE = MODELS / "class_names.json"
SIZE = (224, 224)
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_CANDIDATES = [
    # Latest no-augmentation experiment: 79.17% test accuracy in the user's run.
    MODELS / "experiment_noaug_best.keras",
    # Previous checkpoints kept as fallbacks.
    MODELS / "experimental_best_brain_tumor_cnn.keras",
    MODELS / "best_brain_tumor_cnn.keras",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#101513;--panel:#19211d;--line:#34483d;--text:#edf5ef;--muted:#a9b9ae;--green:#36c879;--mint:#8ce6b2}
.stApp{background:radial-gradient(ellipse at 88% 0%,#26392e 0%,#17221c 35%,#101513 78%);color:var(--text);font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#121a16;border-right:1px solid #2d4035}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:#f1f7f2!important;overflow-wrap:anywhere}
p,li,label{color:#d0ddd3}
.hero{padding:8px 0 22px}.eyebrow{color:var(--mint);font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.hero h1{font-size:clamp(30px,3.6vw,48px);line-height:1.12;margin:9px 0 12px}
.hero p{color:var(--muted);max-width:850px;line-height:1.7;font-size:15px}
[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,rgba(31,44,36,.97),rgba(23,32,27,.97));border:1px solid #3a5144!important;border-radius:18px!important;padding:17px 19px!important;box-shadow:0 12px 28px rgba(0,0,0,.16)}
.metric-card{background:linear-gradient(145deg,#26372d,#1a261f);border:1px solid #405b49;border-radius:15px;padding:16px;height:126px;min-height:126px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:flex-start;overflow:hidden}
.metric-label{color:#b7cbbd;font-size:10px;letter-spacing:1.2px;text-transform:uppercase}
.metric-value{color:#f3fff6;font:700 23px 'Space Grotesk',sans-serif;margin-top:7px;line-height:1.15;overflow-wrap:break-word;word-break:normal;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.metric-note{color:#8ce6b2;font-size:11px;line-height:1.3;margin-top:6px;overflow-wrap:anywhere}
.pill{display:inline-block;padding:5px 10px;border:1px solid #397b56;background:#1b3828;border-radius:999px;color:#aaf0c4;font-size:10px;font-weight:700;letter-spacing:1px}
[data-testid="stSidebar"] div.stButton>button{background:#1c2921!important;color:#e8f5eb!important;border:1px solid #344b3c!important;border-radius:11px!important;text-align:left!important;min-height:44px;font-weight:600}
[data-testid="stSidebar"] div.stButton>button:hover{background:#294333!important;border-color:#55b77a!important;color:#fff!important}
.stApp div.stButton>button[kind="primary"],.stApp div.stDownloadButton>button{background:linear-gradient(100deg,#25a85e,#62d992)!important;color:#082313!important;border:1px solid #78e3a1!important;border-radius:11px!important;font-weight:800;min-height:43px}
.stApp div.stButton>button:not([kind="primary"]){background:#26372d!important;color:#effaf2!important;border:1px solid #48604f!important;border-radius:10px!important}
div[data-testid="stFileUploader"]{background:#19251e;border:1px dashed #5d9c70;border-radius:14px;padding:12px}
[data-testid="stTextInput"] input{background:#111a14;color:#f3fff6;border:1px solid #45604d;border-radius:9px}
[data-testid="stMetric"]{background:#202d24;border:1px solid #405844;padding:13px;border-radius:12px}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#209b55,#8ce6b2)}
hr{border-color:#34483d}
</style>
""", unsafe_allow_html=True)

def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def class_names():
    if CLASS_FILE.exists():
        try:
            raw = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list): names = [str(x) for x in raw]
            elif isinstance(raw, dict) and all(str(k).isdigit() for k in raw):
                names = [str(raw[str(i)] if str(i) in raw else raw[i]) for i in range(len(raw))]
            elif isinstance(raw, dict): names = [str(k) for k, _ in sorted(raw.items(), key=lambda x: x[1])]
            else: names = DEFAULT_CLASSES
            if len(names) == 4: return names
        except Exception: pass
    return DEFAULT_CLASSES

def prediction_model():
    return next((p for p in MODEL_CANDIDATES if p.is_file()), None)

@st.cache_resource(show_spinner="Loading trained CNN...")
def load_model(path, mtime):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)

def preprocess(image):
    original = ImageOps.exif_transpose(image).convert("RGB")
    resized = original.resize(SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return original, resized, arr

def predict(image):
    path = prediction_model()
    if path is None: raise FileNotFoundError("Add models/experimental_best_brain_tumor_cnn.keras or models/best_brain_tumor_cnn.keras.")
    model = load_model(str(path.resolve()), path.stat().st_mtime)
    original, resized, arr = preprocess(image)
    scores = np.asarray(model.predict(arr[None, ...], verbose=0))[0].astype(float).ravel()
    if len(scores) != 4: raise ValueError(f"Expected four outputs; received {len(scores)}.")
    if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=.03):
        exp = np.exp(scores - np.max(scores)); scores = exp / exp.sum()
    names = class_names()
    if len(names) != len(scores): raise ValueError("class_names.json does not match model outputs.")
    idx = int(np.argmax(scores))
    return {"label": names[idx], "confidence": float(scores[idx]), "labels": names,
            "scores": scores.tolist(), "model_path": str(path), "original": original,
            "resized": resized, "array": arr}

def gradcam(image_array, model_path, class_index):
    import tensorflow as tf
    path = Path(model_path)
    model = load_model(str(path.resolve()), path.stat().st_mtime)
    try: conv = model.get_layer("block3_conv2")
    except ValueError as exc:
        raise ValueError("Grad-CAM expects the project's block3_conv2 layer.") from exc
    grad_model = tf.keras.Model(model.inputs, [conv.output, model.output])
    x = tf.convert_to_tensor(image_array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        features, outputs = grad_model(x, training=False)
        score = outputs[:, class_index]
    grads = tape.gradient(score, features)
    if grads is None: raise RuntimeError("Gradients unavailable for this checkpoint.")
    weights = tf.reduce_mean(grads, axis=(1, 2))[0]
    heat = tf.reduce_sum(features[0] * weights, axis=-1)
    heat = tf.maximum(heat, 0)
    maximum = tf.reduce_max(heat)
    heat = tf.where(maximum > 0, heat / maximum, heat)
    return heat.numpy()

def gradcam_overlay(image, heatmap, alpha=.38):
    import matplotlib.cm as cm
    heat_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(image.size)
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0
    colored = cm.get_cmap("jet")(heat)[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8(np.clip((1-alpha)*base + alpha*colored, 0, 1)*255))


def gradcam_attention_view(image, heatmap, threshold=0.60, alpha=0.48):
    """Create a thresholded high-attention view; this is NOT tumor segmentation."""
    import matplotlib.cm as cm
    from PIL import ImageDraw, ImageFilter

    resized = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(resized, dtype=np.float32) / 255.0
    mask = (heat >= float(threshold)).astype(np.uint8) * 255

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    # Tint high-activation pixels in red so the selected regions are easier to distinguish.
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    active = mask.astype(bool)
    blended = base.copy()
    blended[active] = (1-alpha) * base[active] + alpha * red[active]
    result = Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))

    # Draw an outline around thresholded regions, without claiming they are lesions.
    mask_img = Image.fromarray(mask).filter(ImageFilter.MaxFilter(5))
    eroded = Image.fromarray(mask).filter(ImageFilter.MinFilter(5))
    outline = np.asarray(mask_img) > np.asarray(eroded)
    draw = ImageDraw.Draw(result)
    ys, xs = np.where(outline)
    if len(xs):
        # Pixel-wise outline is clear even when OpenCV is not installed.
        for x, y in zip(xs[::2], ys[::2]):
            draw.point((int(x), int(y)), fill=(255, 255, 255))
    return result, mask, float(active.mean()) * 100.0

def pdf_report(result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NVTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#17472c"), alignment=1))
    styles.add(ParagraphStyle(name="NVNote", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#526477")))
    styles.add(ParagraphStyle(name="NVSection", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#16804a"), spaceBefore=10, spaceAfter=6))
    generated = datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z")
    story = [Paragraph("NEUROVISION AI", styles["NVTitle"]),
             Paragraph("Brain MRI Classification — Prediction Summary", styles["Heading2"]),
             Spacer(1, 4*mm), HRFlowable(width="100%", thickness=1, color=colors.HexColor("#36a968")),
             Spacer(1, 4*mm), Paragraph("<b>ACADEMIC RESEARCH OUTPUT — NOT A MEDICAL DIAGNOSIS</b>", styles["Heading3"]),
             Paragraph("This report records an experimental image-classification output. It is not clinically validated and must not be used to diagnose, treat, or rule out a medical condition. A qualified healthcare professional must interpret the MRI.", styles["NVNote"]),
             Paragraph("Patient / scan details", styles["NVSection"])]
    details = [["Patient name", result.get("patient") or "Not provided", "Reference ID", result.get("reference") or "Not provided"],
               ["Image file", result.get("filename") or "Not recorded", "Generated", generated]]
    t = Table(details, colWidths=[30*mm,57*mm,27*mm,58*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#eff7f1")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d1e2d6")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),7)]))
    story += [t, Paragraph("Model output", styles["NVSection"])]
    p = Table([["Highest-scoring class", result["label"].replace("_"," ").title()],["Model score",f'{result["confidence"]:.2%}']],colWidths=[65*mm,107*mm])
    p.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#eaf7ee")),("GRID",(0,0),(-1,-1),.5,colors.HexColor("#b8d9c2")),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("PADDING",(0,0),(-1,-1),8)]))
    story += [p, Paragraph("Scores for all model classes", styles["NVSection"])]
    rows = [["Class","Model score"]] + [[str(n).replace("_"," ").title(),f"{float(s):.2%}"] for n,s in sorted(zip(result["labels"],result["scores"]),key=lambda x:float(x[1]),reverse=True)]
    tab = Table(rows,colWidths=[110*mm,62*mm])
    tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17472c")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d1e2d6")),("FONTSIZE",(0,0),(-1,-1),9),("PADDING",(0,0),(-1,-1),7)]))
    story += [tab, Paragraph("Technical notes",styles["NVSection"]),
              Paragraph(f'Model file: {result["model_path"]}<br/>Preprocessing: RGB, 224 × 224, pixel scaling to 0–1.<br/>Scores are raw model outputs and may not be calibrated probabilities.',styles["NVNote"]),
              Spacer(1,5*mm),HRFlowable(width="100%",thickness=.7,color=colors.HexColor("#d1e2d6")),Spacer(1,3*mm),
              Paragraph("This is not a radiology report or substitute for review by a qualified clinician.",styles["NVNote"])]
    doc.build(story); buf.seek(0); return buf.getvalue()

def load_json(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None

def extract_metrics(data):
    if not isinstance(data,dict): return {}
    out={}
    for k in ("test_accuracy","accuracy"):
        if k in data: out["Accuracy"]=float(data[k]); break
    report=data.get("classification_report",{})
    macro=report.get("macro avg",report.get("macro_avg",{})) if isinstance(report,dict) else {}
    for src,dst in (("precision","Macro precision"),("recall","Macro recall"),("f1-score","Macro F1"),("f1_score","Macro F1")):
        if isinstance(macro,dict) and src in macro: out[dst]=float(macro[src])
    for src,dst in (("macro_precision","Macro precision"),("macro_recall","Macro recall"),("macro_f1","Macro F1")):
        if src in data: out[dst]=float(data[src])
    out["Test samples"]=data.get("test_samples",data.get("test_images","—"))
    return out

def metric_file(paths):
    for path in paths:
        if path.is_file():
            vals=extract_metrics(load_json(path))
            if vals: return path,vals
    return None,{}


def read_prediction_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_prediction_history(items):
    REPORTS.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(items[:500], indent=2), encoding="utf-8")

def overview():
    hero("NEUROVISION AI / RESEARCH WORKSPACE","Brain MRI intelligence, clearly presented.","A student-built custom CNN prototype for four-category brain MRI classification, with explainability, experiment review, and downloadable prediction summaries.")
    model=prediction_model()
    metric_source,vals=metric_file([REPORTS/"experimental_test_metrics.json",REPORTS/"original_metrics.json",REPORTS/"test_metrics.json",REPORTS/"evaluation_test_metrics.json"])
    a,b,c,d=st.columns(4)
    with a: card("Prediction model","Custom CNN" if model else "Not found","Baseline checkpoint")
    with b: card("Input","224 × 224","RGB · normalized 0–1")
    with c: card("Categories","04","Dataset-defined labels")
    with d: card("Latest no-augmentation accuracy" if metric_source and metric_source.name == "experimental_test_metrics.json" else "Saved test accuracy", f'{vals["Accuracy"]:.2%}' if "Accuracy" in vals else "—", f'{vals.get("Test samples", "Test set")} test images · {metric_source.name}' if metric_source else "Run evaluation to populate")
    l,r=st.columns([1.2,.8],gap="large")
    with l:
        with st.container(border=True):
            st.markdown("### Start an analysis")
            st.write("Upload an MRI image, run the CNN, inspect class scores and Grad-CAM, then export a PDF summary.")
            if st.button("Open MRI Analyzer",type="primary",use_container_width=True): st.session_state.page="MRI Analyzer"; st.rerun()
    with r:
        with st.container(border=True):
            st.markdown("### Model categories")
            for name in class_names(): st.write("• "+name.replace("_"," ").title())
            st.caption("Classification only: no segmentation, grading, or clinical diagnosis.")

def analyzer():
    hero("INTERACTIVE ANALYSIS", "MRI Analyzer",
         "Upload an MRI image, run the selected CNN, inspect all four class scores, generate Grad-CAM, and export a structured PDF summary.")
    model_path = prediction_model()
    model_display = "No-augmentation CNN" if model_path and model_path.name == "experiment_noaug_best.keras" else (model_path.stem.replace("_", " ").title() if model_path else "Not found")
    model_note = f"Checkpoint: {model_path.name}" if model_path else "Place a .keras checkpoint in models/"
    a, b, c = st.columns(3, gap="medium")
    with a: card("Active checkpoint", model_display, model_note)
    with b: card("Input specification", "224 × 224", "RGB · scaled to 0–1")
    with c: card("Output classes", "04", "Glioma · Meningioma · No tumor · Pituitary")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### 01 · Patient and scan details")
            st.caption("Details are optional. Prefer a non-identifying patient reference.")
            patient = st.text_input("Patient name (optional)", key="analyzer_patient")
            reference = st.text_input("Patient / scan reference", key="analyzer_reference", placeholder="e.g. NV-2026-001")
            upload = st.file_uploader("Upload brain MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="analyzer_upload")
            if upload:
                try:
                    preview = ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert("RGB")
                    st.image(preview, caption=f"Preview · {preview.width} × {preview.height} px", use_container_width=True)
                    st.caption("The model resizes this image to 224 × 224 RGB before inference.")
                except Exception:
                    st.error("This image could not be opened.")
                    upload = None
            run = st.button("Run MRI classification", type="primary", use_container_width=True, disabled=(upload is None or model_path is None))

    with right:
        with st.container(border=True):
            st.markdown("### 02 · Prediction result")
            if not model_path:
                st.error("No compatible CNN checkpoint was found in models/.")
            if run and upload:
                try:
                    with st.spinner("Loading CNN and analyzing image..."):
                        pred = predict(Image.open(io.BytesIO(upload.getvalue())))
                    pred.update({"filename": upload.name, "patient": patient.strip(), "reference": reference.strip(),
                                 "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")})
                    compact = {k: pred[k] for k in ("label", "confidence", "labels", "scores", "filename", "patient", "reference", "model_path", "timestamp")}
                    compact["record_id"] = f'{compact["timestamp"]}|{compact["filename"]}'
                    st.session_state.last_result = compact
                    st.session_state.last_array = pred["array"]
                    st.session_state.last_preview = pred["resized"]
                    st.session_state.gradcam_image = None
                    history = read_prediction_history()
                    history.insert(0, compact)
                    save_prediction_history(history)
                    st.session_state.history = read_prediction_history()
                    st.success("Analysis completed and prediction metadata saved locally.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            result = st.session_state.get("last_result")
            if result:
                st.markdown('<span class="pill">MODEL OUTPUT · EXPERIMENTAL</span>', unsafe_allow_html=True)
                st.caption(f"Last analyzed image: {result.get('filename', 'unknown')} · {result.get('timestamp', '')}")
                if st.button("Clear displayed result", key="clear_analyzer_result"):
                    for key in ("last_result", "last_array", "last_preview", "gradcam_image", "gradcam_heatmap"):
                        st.session_state.pop(key, None)
                    st.rerun()
                st.markdown(f"## {result['label'].replace('_', ' ').title()}")
                st.metric("Highest model score", f"{result['confidence']:.2%}")
                st.caption("Highest model output score; not a calibrated probability or diagnosis.")
                score_df = pd.DataFrame({"Class": [x.replace("_", " ").title() for x in result["labels"]], "Score": result["scores"]}).sort_values("Score", ascending=False)
                st.markdown("#### Class score breakdown")
                for _, row in score_df.iterrows():
                    st.markdown(f"**{row['Class']}** · {row['Score']:.2%}")
                    st.progress(float(max(0, min(1, row["Score"]))))
                st.caption(f"Checkpoint: {Path(result['model_path']).name}")
                try:
                    pdf = pdf_report(result)
                    safe = "".join(ch for ch in (result.get("reference") or "scan") if ch.isalnum() or ch in "-_")
                    st.download_button("Download patient summary (PDF)", pdf, f"NeuroVision_Report_{safe}.pdf", "application/pdf", type="primary", use_container_width=True)
                except ImportError:
                    st.error("Install PDF dependency: python -m pip install reportlab")
            else:
                st.info("Upload an MRI image on the left, then select Run MRI classification.")
                st.write("The result panel will show the predicted category, all four class scores, active checkpoint and downloadable PDF.")

    result = st.session_state.get("last_result")
    if result:
        with st.container(border=True):
            st.markdown("### 03 · Explainability — Grad-CAM")
            st.write("Grad-CAM highlights feature-map regions influencing the selected class. It is not a tumor boundary or clinical localization.")
            if st.button("Generate Grad-CAM visualization", use_container_width=True):
                try:
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        idx = result["labels"].index(result["label"])
                        heatmap = gradcam(st.session_state.last_array, result["model_path"], idx)
                        st.session_state.gradcam_heatmap = heatmap
                        st.session_state.gradcam_image = gradcam_overlay(st.session_state.last_preview, heatmap)
                except Exception as e:
                    st.error(f"Grad-CAM could not be generated: {e}")

            heatmap = st.session_state.get("gradcam_heatmap")
            if heatmap is not None:
                threshold = st.slider(
                    "High-attention threshold",
                    min_value=0.20, max_value=0.90, value=0.60, step=0.05,
                    help="Raises the cutoff for highlighting stronger Grad-CAM activations. This is a visualization setting, not a medical threshold."
                )
                attention_view, attention_mask, highlighted_pct = gradcam_attention_view(
                    st.session_state.last_preview, heatmap, threshold=threshold
                )
                x, y = st.columns(2)
                with x:
                    st.image(st.session_state.last_preview, caption="Original model input", use_container_width=True)
                with y:
                    st.image(st.session_state.gradcam_image, caption="Grad-CAM heatmap overlay", use_container_width=True)
                st.markdown("#### High-attention region view")
                st.image(attention_view, caption=f"Pixels above visualization threshold ({threshold:.0%}) are outlined/tinted red", use_container_width=True)
                st.metric("Image area above selected threshold", f"{highlighted_pct:.1f}%")
                st.caption("This percentage describes the area highlighted by the Grad-CAM threshold only. It is not the percentage of the brain affected by a tumor.")
                with st.expander("How to read the highlighted regions"):
                    st.write(
                        "Red/tinted areas have stronger relative influence on the selected class score at the chosen threshold. "
                        "Blue/low-activation areas contribute less to this particular Grad-CAM map. "
                        "A highlighted area is not automatically an abnormality: the current classifier was trained for image-level "
                        "categories, not pixel-level tumor segmentation. To identify actual tumor boundaries, you would need "
                        "pixel-level lesion masks and a segmentation model validated against expert annotations."
                    )
    st.warning("Academic prototype only — not a medical device. Do not use predictions for diagnosis or treatment. MRI findings must be interpreted by a qualified clinician.")


def history_page():
    hero("ANALYSIS LOG", "Prediction history",
         "Locally saved prediction records. Only prediction metadata is stored; uploaded MRI images are not saved.")
    hist = read_prediction_history()
    st.session_state.history = hist
    if not hist:
        st.info("No saved predictions yet. Run a classification in MRI Analyzer. Completed predictions will remain available after restarting the app.")
        if st.button("Open MRI Analyzer", type="primary"):
            st.session_state.page = "MRI Analyzer"
            st.rerun()
        return
    st.caption(f"{len(hist)} saved record(s) · reports/prediction_history.json")
    rows = [{"Date / time": x.get("timestamp", ""), "Reference": x.get("reference") or "—",
             "Patient": x.get("patient") or "—", "Image": x.get("filename", ""),
             "Prediction": x.get("label", "").replace("_", " ").title(),
             "Model score": f'{float(x.get("confidence", 0)):.2%}',
             "Model": Path(x.get("model_path", "")).name} for x in hist]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Export history CSV", df.to_csv(index=False).encode("utf-8"), "neurovision_prediction_history.csv", "text/csv")
    with st.expander("View an individual saved prediction"):
        selected = st.selectbox("Select record", list(range(len(hist))),
            format_func=lambda i: f'{hist[i].get("timestamp","")} · {hist[i].get("filename","")}')
        item = hist[selected]
        st.markdown(f"### {item.get('label','').replace('_',' ').title()}")
        st.write(f"Highest model score: **{float(item.get('confidence',0)):.2%}**")
        detail = pd.DataFrame({"Class": [x.replace("_", " ").title() for x in item.get("labels", [])],
                               "Score": [f"{float(x):.2%}" for x in item.get("scores", [])]})
        st.dataframe(detail, use_container_width=True, hide_index=True)
        try:
            st.download_button("Download this saved PDF report", pdf_report(item), f"NeuroVision_Report_{selected+1}.pdf", "application/pdf")
        except Exception as e:
            st.caption(f"PDF report unavailable: {e}")
    if st.button("Clear all saved prediction history"):
        save_prediction_history([])
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.gradcam_image = None
        st.success("Local prediction history cleared.")
        st.rerun()


def comparison():
    hero("MODEL EVALUATION","Model comparison studio","Compare locally saved evaluation JSON files. Missing metrics are left blank rather than invented.")
    candidates={
      "Original / baseline":[REPORTS/"original_metrics.json",REPORTS/"test_metrics.json",REPORTS/"evaluation_test_metrics.json"],
      "Fine-tuned":[REPORTS/"finetuned_test_metrics.json",REPORTS/"experimental_test_metrics.json"],
      "No-augmentation":[
          REPORTS/"experimental_test_metrics.json",
          REPORTS/"noaug_test_metrics.json",
          REPORTS/"experiment_noaug_test_metrics.json"
      ],
    }
    rows=[]
    for name,paths in candidates.items():
        path,values=metric_file(paths)
        if path:
            row={"Model":name,"Source":path.name,**values}; rows.append(row)
    if not rows: st.info("No compatible metrics JSON files found in reports/.")
    else:
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        if {"Original / baseline", "Fine-tuned"}.issubset(set(df["Model"])):
            base = df.loc[df["Model"] == "Original / baseline"].iloc[0]
            fine = df.loc[df["Model"] == "Fine-tuned"].iloc[0]
            compare_cols = [c for c in ("Accuracy", "Macro precision", "Macro recall", "Macro F1") if c in df.columns]
            if compare_cols and all(
                pd.isna(base[c]) and pd.isna(fine[c]) or
                (pd.notna(base[c]) and pd.notna(fine[c]) and abs(float(base[c]) - float(fine[c])) < 1e-9)
                for c in compare_cols
            ):
                st.warning("Baseline and fine-tuned metrics are identical. Check that original_metrics.json and finetuned_test_metrics.json were generated from their respective model checkpoints and the same test split.")
        chart_cols=[c for c in ("Accuracy","Macro precision","Macro recall","Macro F1") if c in df.columns]
        if chart_cols:
            chart = df.set_index("Model")[chart_cols].copy()
            # Metrics are stored as fractions (0-1); display as percentage points.
            chart = chart * 100
            st.markdown("### Metric comparison")
            st.caption("Grouped bars show each metric independently; values are percentages.")
            st.bar_chart(chart, stack=False, height=360)
        st.caption("Only compare models evaluated on the same held-out test split and preprocessing. Do not tune repeatedly against the test set.")
    st.markdown("### Saved confusion matrices")
    if FIGURES.exists():
        for path in sorted(FIGURES.glob("*confusion_matrix*.png")):
            with st.expander(path.name): st.image(str(path),use_container_width=True)
    else: st.info("No reports/figures folder found.")

def training_graphs():
    hero("TRAINING MONITOR","Training curves","Review accuracy and loss histories saved by your training scripts.")
    paths=[
        REPORTS/"experiment_noaug_history.csv",
        REPORTS/"experimental_training_history.csv",
        REPORTS/"training_history.csv"
    ]
    available=[p for p in paths if p.is_file()]
    if not available:
        st.info("No training-history CSV found. Expected reports/experiment_noaug_history.csv or reports/experimental_training_history.csv.")
    else:
        chosen=st.selectbox("Choose training run",available,format_func=lambda p:p.name)
        df=pd.read_csv(chosen)
        if not df.empty:
            df.index=np.arange(1,len(df)+1)
            a,b=st.columns(2)
            with a:
                st.markdown("### Accuracy")
                cols=[c for c in ("accuracy","val_accuracy") if c in df]
                if cols: st.line_chart(df[cols].rename(columns={"accuracy":"Training","val_accuracy":"Validation"}))
                else: st.info("No accuracy columns found.")
            with b:
                st.markdown("### Loss")
                cols=[c for c in ("loss","val_loss") if c in df]
                if cols: st.line_chart(df[cols].rename(columns={"loss":"Training","val_loss":"Validation"}))
                else: st.info("No loss columns found.")
            st.dataframe(df.reset_index(names="Epoch"),use_container_width=True,hide_index=True)
    for p in (FIGURES/"experimental_training_curves.png",FIGURES/"noaug_training_curves.png"):
        if p.is_file(): st.image(str(p),caption=p.name,use_container_width=True)

def workflow():
    hero("METHODOLOGY","AI workflow","The end-to-end pipeline used by the custom CNN.")
    steps=[("01","Dataset","Labelled MRI images are separated into training, validation, and held-out test splits."),
           ("02","Preprocessing","Images are converted to RGB, resized to 224 × 224, and scaled to 0–1."),
           ("03","Augmentation","Training-only rotations, zoom and contrast changes can be enabled."),
           ("04","Feature extraction","Three convolutional blocks use 32, 64 and 128 filters."),
           ("05","Classification","Global Average Pooling and dense layers produce four class scores."),
           ("06","Evaluation","Accuracy, precision, recall, F1-score and confusion matrix summarize held-out performance."),
           ("07","Explainability","Grad-CAM visualizes influential feature regions; it is not segmentation.")]
    for n,title,desc in steps:
        with st.container(border=True):
            st.markdown(f'<div class="eyebrow">STEP {n}</div>',unsafe_allow_html=True); st.subheader(title); st.write(desc)

def about():
    hero("PROJECT INFORMATION","About NeuroVision AI","A student research prototype exploring custom CNN-based brain MRI classification.")
    with st.container(border=True):
        st.markdown("### Objective"); st.write("Classify MRI images into glioma, meningioma, no tumor, and pituitary dataset categories.")
        st.markdown("### Technology"); st.write("Python · TensorFlow/Keras · Streamlit · Pillow · NumPy · Pandas · Matplotlib · Scikit-learn · ReportLab")
        st.markdown("### Scope"); st.write("Image classification only. The application does not perform tumor segmentation, localization, grading, or clinical diagnosis.")
        st.warning("Academic prototype only. Predictions must not be used to guide medical care.")

if "page" not in st.session_state: st.session_state.page="Overview"
if "last_result" not in st.session_state: st.session_state.last_result=None
if "history" not in st.session_state: st.session_state.history=read_prediction_history()
if "gradcam_image" not in st.session_state: st.session_state.gradcam_image=None

NAV=[("Overview","⌂"),("MRI Analyzer","◉"),("Prediction History","▤"),("Model Comparison","▥"),("Training Graphs","⌁"),("AI Workflow","⇢"),("About Project","ⓘ")]
with st.sidebar:
    st.markdown("## 🧠 NeuroVision AI")
    st.markdown('<span class="pill">CHARCOAL / EMERALD</span>',unsafe_allow_html=True)
    st.caption("BRAIN MRI · DEEP LEARNING"); st.divider()
    for name,icon in NAV:
        if st.button(f"{icon}  {name}",key="nav_"+name,use_container_width=True):
            st.session_state.page=name; st.rerun()
    st.divider()
    st.success("Baseline CNN found") if prediction_model() else st.warning("Model checkpoint not found")
    st.caption("Local TensorFlow / Keras inference")
    st.caption("History is session-only.")

page=st.session_state.page
if page=="Overview": overview()
elif page=="MRI Analyzer": analyzer()
elif page=="Prediction History": history_page()
elif page=="Model Comparison": comparison()
elif page=="Training Graphs": training_graphs()
elif page=="AI Workflow": workflow()
elif page=="About Project": about()
st.divider()
st.caption("NEUROVISION AI · ACADEMIC PROJECT PROTOTYPE · NOT FOR CLINICAL DIAGNOSIS")
