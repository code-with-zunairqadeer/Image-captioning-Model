"""Core image captioning logic, powered by Salesforce BLIP.

BLIP (Bootstrapping Language-Image Pre-training) is a vision-language
model: it looks at the image with a vision transformer and writes a
caption with a language decoder. We use the public pre-trained weights
from Hugging Face - no training needed.
"""

from PIL import Image

_model = None
_processor = None

MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_model():
    """Load BLIP once and reuse it (downloads ~1GB on first run)."""
    global _model, _processor
    if _model is None:
        # Imported here so the module stays light until needed.
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    return _processor, _model


def caption_image(image_path, max_length=50):
    """Generate a caption for the image at *image_path*.

    Returns the caption string.
    """
    processor, model = load_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
