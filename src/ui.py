import gradio as gr
import json, ast, os, tempfile
import pandas as pd
from PIL import Image as PILImage
from src.utils import encode_image_to_base64, find_similar_items
from src.main import analyze_image, get_embeddings, check_match

IMAGES_DIR = os.path.abspath("data/sample_clothes/sample_images")

# Load CSS from external file
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
    with open(css_path, 'r') as f:
        return f.read()

CSS = load_css()

def get_unique_subcategories():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    if not os.path.exists(styles_filepath):
        return []
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    return styles_df['articleType'].unique().tolist()

SUBCATEGORIES = get_unique_subcategories()

def analyze_step(image_np):
    """Step 1: analyze the image, stash the JSON, and return it."""
    if image_np is None:
        # keep the Recommend button hidden if nothing uploaded
        return "👉 Please upload an image first.", None, gr.update(visible=False)
    # save & encode
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        b64 = encode_image_to_base64(tmp.name)
    raw = analyze_image(b64, SUBCATEGORIES)
    # show the Recommend button now
    return raw, raw, gr.update(visible=True)

def recommend_step(analysis_json):
    """Step 2: read the stashed JSON, run RAG, and return image paths."""
    if analysis_json is None:
        return [], gr.update(visible=False)
    info = json.loads(analysis_json)
    descs = info['items']
    gender = info['gender']
    category = info['category']

    styles_fp = "data/sample_clothes/sample_styles_with_embeddings.csv"
    if not os.path.exists(styles_fp):
        return [], gr.update(visible=False)
    df = pd.read_csv(styles_fp, on_bad_lines='skip')
    df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    df = df[df['gender'].isin([gender, 'Unisex'])]
    df = df[df['articleType'] != category]

    # compute similarity
    emb_list = df['embeddings'].tolist()
    similar_indices = []
    for d in descs:
        ie = get_embeddings([d])
        similar_indices += find_similar_items(ie, emb_list)

    gallery_items = []
    for idx in similar_indices:  # Process all similar items
        pid = df.iloc[idx]['id']
        img_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
        if os.path.exists(img_path):
            caption = f"Product {pid}"
            gallery_items.append((img_path, caption))
    if not gallery_items:
        return [], gr.update(visible=False)
    return gallery_items, gr.update(visible=True)

def validate_matches_step(image_np, gallery_paths):
    if image_np is None or not gallery_paths:
        return []
    # Save and encode reference image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image_np).save(tmp.name)
        reference_b64 = encode_image_to_base64(tmp.name)
    validated = []
    reasons = []
    for img_path in gallery_paths:
        # Handle both string paths and tuples containing paths
        if isinstance(img_path, tuple):
            img_path = img_path[0]  # Take the first element if it's a tuple
        suggested_b64 = encode_image_to_base64(img_path)
        result = check_match(reference_b64, suggested_b64)
        if hasattr(result, 'answer') and result.answer.lower() == "yes":
            validated.append(img_path)
            reasons.append(f"{getattr(result, 'reason', '')}")
        elif isinstance(result, dict) and result.get('answer', '').lower() == "yes":
            validated.append(img_path)
            reasons.append(f"{result.get('reason', '')}")
    if not validated:
        return []

    # build list of (img, caption) for gallery
    gallery_items = list(zip(validated, reasons))
    return gallery_items

# —— Build the UI ——
with gr.Blocks(css=CSS, theme="light") as demo:
    # HEADER & Branding
    gr.HTML("""
      <div id="header">
        <h1>👗 GPT4o Clothing Style Analyzer</h1>
        <p>Discover, recommend, and validate fashion looks with AI. Powered by GPT-4o-mini.</p>
      </div>
    """)
    # Stepper
    gr.HTML("""
      <div class="stepper">
        <div class="step">1. Analyze</div>
        <div class="step inactive">2. Recommend</div>
        <div class="step inactive">3. Validate</div>
      </div>
    """)

    # === Step 1 card ===
    with gr.Row(elem_id="main-card"):
        with gr.Column(scale=6):
            gr.Markdown("""
            ### 1️⃣ Upload Product Image
            Upload a clothing product image for instant AI-powered analysis.
            """)
            img = gr.Image(type="numpy", label="Upload Product Image", elem_id="upload-image")
            analyze_btn = gr.Button("🔍 Analyze the Look", variant="primary")
        with gr.Column(scale=6):
            analysis_txt = gr.Textbox(
                label="Analysis Output (JSON)",
                interactive=False, lines=8,
                placeholder="Your JSON will appear here…"
            )

    analysis_state = gr.State()

    # === Step 2 card ===
    with gr.Row(elem_id="main-card"):
        with gr.Column():
            gr.Markdown("""
            ### 2️⃣ AI Recommendation
            Click below to get AI-powered product recommendations matching your style.
            """)
            recommend_btn = gr.Button("🛍️ Recommend Matching Products", visible=False, variant="primary")
            gallery = gr.Gallery(
                label="Matching Items",
                elem_id="rec_gallery",
                columns=4, 
                height="320px",
                interactive=False,
            )

    # === Step 3 card ===
    with gr.Row(elem_id="main-card"):
        with gr.Column():
            gr.Markdown("""
            ### 3️⃣ Validate Outfit Matches
            Review the recommended matches and validate the best outfit combinations.
            """)
            validate_btn = gr.Button("✅ Validate Outfit Matches", visible=False, variant="primary")
            validated_gallery = gr.Gallery(
                label="Validated Matches",
                columns=2,
                height="320px",
                interactive=False,
            )

    # wiring
    analyze_btn.click(
        fn=analyze_step,
        inputs=[img],
        outputs=[analysis_txt, analysis_state, recommend_btn],
        show_progress=True
    )
    recommend_btn.click(
        fn=recommend_step,
        inputs=[analysis_state],
        outputs=[gallery,validate_btn],
        show_progress=True
    )
    validate_btn.click(
        fn=validate_matches_step,
        inputs=[img, gallery],
        outputs=[validated_gallery],
        show_progress=True
    )


demo.launch(allowed_paths=[IMAGES_DIR])


