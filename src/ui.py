import gradio as gr
import json
import os
from src.utils import encode_image_to_base64
from src.main import analyze_image
import pandas as pd

def get_unique_subcategories():
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    if not os.path.exists(styles_filepath):
        return []
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    return styles_df['articleType'].unique().tolist()

subcategories = get_unique_subcategories()

def analyze_uploaded_image(image):
    if image is None:
        return "Please upload an image."
    # Gradio gives a numpy array, save to temp file
    import tempfile
    from PIL import Image as PILImage
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        PILImage.fromarray(image).save(tmp.name)
        encoded_image = encode_image_to_base64(tmp.name)
    try:
        result = analyze_image(encoded_image, subcategories)
        try:
            result_json = json.loads(result)
            return json.dumps(result_json, indent=2)
        except Exception:
            return result
    except Exception as e:
        return f"Error: {str(e)}"

description = "Upload a clothing image to analyze its style, category, and gender using GPT-4o-mini."
iface = gr.Interface(
    fn=analyze_uploaded_image,
    inputs=gr.Image(type="numpy", label="Upload Clothing Image"),
    outputs=gr.Textbox(label="Analysis Output (JSON)"),
    title="Clothing Image Analyzer",
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()