import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

import csv
import traceback
import cv2
from moviepy import VideoFileClip
import whisper
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    pipeline, logging as hf_logging
)

hf_logging.set_verbosity_error()
from PIL import Image
from rake_nltk import Rake
import nltk
from deep_translator import GoogleTranslator
import torch

# ================= CONFIG =================
VIDEO_FOLDER = "videos"
OUTPUT_CSV = "video_metadata.csv"

WHISPER_MODEL_SIZE = "small"                # Perfect middle-ground for transcription
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"    # Perfect middle-ground LLM (1.5 Billion params)
#WHISPER_MODEL_SIZE = "large-v3"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
VISION_MODEL = "Salesforce/blip-image-captioning-base"
CONTEXT_MODEL = "facebook/bart-large-mnli"
#LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

CONTEXT_LABELS = [
    "Education", "Entertainment", "News", "Technology",
    "Comedy", "Sports", "Politics", "Music", "Gaming",
    "Vlog", "How-to & Style", "Science"
]

# ================= DEVICE =================
if torch.cuda.is_available():
    DEVICE = "cuda"
   # torch.cuda.set_per_process_memory_fraction(0.95)  # limit VRAM to 80% (~9.8GB of 12GB)
   # print("🚀 Using GPU (VRAM capped at 95%)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using Apple GPU")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU")

# ================= INIT =================
print("Loading models...")

# NLTK
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
    except:
        nltk.download(pkg, quiet=True)

# Whisper
audio_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)

# Summarizer
summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZATION_MODEL
).to(DEVICE)

# Vision (BLIP)
vision_processor = BlipProcessor.from_pretrained(VISION_MODEL)
vision_model = BlipForConditionalGeneration.from_pretrained(
    VISION_MODEL
).to(DEVICE)

# Classifier
pipe_device = 0 if DEVICE == "cuda" else -1
classifier = pipeline("zero-shot-classification", model=CONTEXT_MODEL, device=pipe_device)

# LLM (FIXED)
print("Loading LLM...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)

if any(x in LLM_MODEL.lower() for x in ["t5", "bart"]):
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
else:
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )

rake = Rake()

print("✅ All models loaded\n")

# ================= FUNCTIONS =================

def extract_audio(video_path, audio_path="temp.wav"):
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(audio_path, logger=None)
            return audio_path
    except:
        return None
    finally:
        if 'video' in locals():
            video.close()


def extract_frames(video_path, num=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // (num + 1))

    for i in range(1, num + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, step * i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def generate_visual_description(frames):
    desc = []
    for img in frames:
        inputs = vision_processor(img, return_tensors="pt").to(DEVICE)
        out = vision_model.generate(**inputs, max_new_tokens=50)
        caption = vision_processor.decode(out[0], skip_special_tokens=True)
        desc.append(caption)
    return " ".join(set(desc))


def translate_to_english(text):
    """Translate text to English if needed (for English-only models)."""
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text[:4500])
        return translated if translated else text
    except:
        return text


def extract_tags(text, lang="en"):
    """Extract keywords. Translates non-English text first since RAKE is English-only."""
    work_text = translate_to_english(text) if lang != "en" else text
    try:
        rake.extract_keywords_from_text(work_text)
        return [kw for _, kw in rake.get_ranked_phrases_with_scores()[:5]]
    except:
        return []


def generate_llm_text(prompt):
    messages = [
        {"role": "system", "content": "You are an expert YouTube assistant. Output ONLY the description and hashtags. Do not include markdown blocks, conversational filler, or repeat the prompt."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        formatted_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        formatted_prompt = prompt

    inputs = llm_tokenizer(formatted_prompt, return_tensors="pt")

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )

    if any(x in LLM_MODEL.lower() for x in ["t5", "bart"]):
        generated_ids = outputs[0]
    else:
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]

    return llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def analyze_video(video_path):
    meta = {"filename": os.path.basename(video_path)}

    try:
        # AUDIO
        audio = extract_audio(video_path)
        if audio:
            result = audio_model.transcribe(audio)
            text = result["text"]
            lang = result.get("language", "en")  # e.g. 'hi', 'en', 'te'

            meta["transcription"] = text  # always in original language
            meta["language"] = lang

            # For English-only models: translate non-English text first
            text_en = translate_to_english(text) if lang != "en" else text

            if len(text_en.split()) > 20:
                inputs = summarizer_tokenizer(text_en, return_tensors="pt", truncation=True).to(DEVICE)
                summary_ids = summarizer_model.generate(inputs["input_ids"], forced_bos_token_id=0)
                summary_en = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            else:
                summary_en = text_en

            meta["summary"] = summary_en
            meta["title"] = summary_en[:60]
            meta["english_title"] = meta["title"]  # already in English

            try:
                meta["context"] = classifier(text_en, CONTEXT_LABELS)['labels'][0]
            except:
                meta["context"] = "Unknown"

            os.remove(audio)

        # VISUAL
        frames = extract_frames(video_path)
        meta["visual_content"] = generate_visual_description(frames)

        # LLM
        prompt = f"""Write a catchy YouTube description and hashtags for this video.
Do NOT include the original title, context, or visual info in your output. Just output the description paragraph followed by hashtags. Do not add any conversational text.

Title: {meta.get('english_title')}
Context: {meta.get('context')}
Content: {meta.get('summary')}
Visual: {meta.get('visual_content')}"""

        llm_output = generate_llm_text(prompt)
        meta["description"] = llm_output

        # Extract hashtags from the LLM output to use as metatags (fallback to RAKE)
        hashtags = [w.strip() for w in llm_output.split() if w.startswith("#")]
        if hashtags:
            meta["metatags"] = ", ".join(hashtags)
        else:
            meta["metatags"] = ", ".join(extract_tags(meta.get("summary", ""), lang="en"))

        return meta

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return None


# ================= MAIN =================

def process_all_videos():
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4",".mov",".avi",".mkv"))]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename","language","title","english_title","context",
            "summary","description","transcription","visual_content","metatags"
        ], extrasaction='ignore')
        writer.writeheader()

        for v in videos:
            data = analyze_video(os.path.join(VIDEO_FOLDER, v))
            if data:
                writer.writerow(data)
                print("✔ Done:", v)


if __name__ == "__main__":
    process_all_videos()