# Image Captioning App 📸 → 📝

Upload any photo, get an AI-written caption. Powered by **Salesforce BLIP**, a vision-language model — no training required, we use public pre-trained weights.

## How it works

BLIP has two halves: a **vision transformer** that "sees" the image, and a **language decoder** that writes the caption. The pre-trained model already learned the link between images and words from millions of image-text pairs — our app just loads it and asks it to describe your photo.

```
your photo → BLIP vision encoder → language decoder → "a dog playing in the park"
```

## Quickstart

Python 3.9+.

```bash
git clone https://github.com/codewithzunair-q/Image-captioning-model-.git
cd Image-captioning-model-
pip install -r requirements.txt
python app.py
```

Open the URL it prints (usually http://127.0.0.1:7860), upload a photo, read the caption.

> **First run downloads ~1GB** of model weights from Hugging Face and caches them locally. Later runs start instantly.

## Project structure

```
image-captioning-app/
├── caption.py          # core logic: load BLIP, caption an image file
├── app.py              # Gradio web UI
├── requirements.txt    # transformers, torch, pillow, gradio
└── README.md
```

## Try it on

- Your own photos — portraits, street scenes, food
- Tricky images — does it catch details? Where does it fail?
- Compare `max_length` values in `caption_image()` — short vs descriptive captions

## Ideas to extend

- Batch-caption a whole folder and save results to CSV
- Add a "confidence" display or alternative captions via beam search
- Swap in `Salesforce/blip-image-captioning-large` for better quality
- Deploy the Gradio app to Hugging Face Spaces (free hosting)

---

*Project 2: from generating Urdu poetry with n-grams to describing images with a vision-language model — the portfolio's "machines that understand" theme continues.*
