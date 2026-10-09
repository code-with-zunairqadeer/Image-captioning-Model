"""Gradio web app: upload an image, get an AI-generated caption.

Usage:
    pip install -r requirements.txt
    python app.py

Then open the URL it prints (usually http://127.0.0.1:7860).
"""

import gradio as gr

from caption import caption_image


def describe(uploaded_path):
    if not uploaded_path:
        return "Please upload an image first."
    try:
        return caption_image(uploaded_path)
    except Exception as exc:  # noqa: BLE001 - show friendly error in UI
        return f"Something went wrong: {exc}"


demo = gr.Interface(
    fn=describe,
    inputs=gr.Image(type="filepath", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Captioning App",
    description=(
        "Upload any photo and get an AI-written caption, "
        "powered by Salesforce BLIP (vision-language model)."
    ),
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
