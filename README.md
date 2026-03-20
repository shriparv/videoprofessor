# VideoProfessor 🎥🤖

VideoProfessor is an advanced, fully local AI pipeline built in Python to automatically analyze videos, extract transcriptions, generate catchy YouTube descriptions, identify contexts, and output polished metadata into a CSV format.

## Features ✨

* **Multi-Modal AI Pipeline**: Combines Audio (Whisper), Vision (BLIP), and Text (Qwen) into a single processing workflow.
* **Auto-Transcription & Language Detection**: Uses OpenAI's `Whisper` to transcribe audio and detect the spoken language.
* **Seamless Multilingual Support (Hindi/Hinglish)**: Automatically translates non-English text to English before feeding it through the English-only classification and summarization models, keeping your original transcription intact.
* **Visual Context Extraction**: Samples frames from the video and uses `Salesforce/blip-image-captioning-base` to generate visual descriptions.
* **YouTube Description & Hashtags Generation**: Uses a local instruction LLM (`Qwen/Qwen2.5-1.5B-Instruct`) to generate smart, catchy descriptions and high-quality hashtags perfectly tailored to the content.
* **Zero-Shot Classification**: Categorizes the video into topics (Education, Entertainment, Tech, etc.) using `facebook/bart-large-mnli`.
* **Hardware Acceleration**: Automatically detects and uses NVIDIA GPUs (`cuda`) or Apple Silicon GPUs (`mps`). Implements VRAM capping to keep your system responsive.
* **Excel-Ready Export**: Saves all extracted data into `video_metadata.csv` using `utf-8-sig` encoding so languages like Hindi render perfectly in Microsoft Excel without garbled text.

## Requirements 📦

The project runs completely locally, meaning you need a machine with a dedicated GPU (e.g. an RTX 4070 or better) and enough VRAM (~6-8GB) to run the models efficiently. 

**Core Dependencies:**
* `torch`, `torchvision`, `torchaudio` (Compiled with CUDA 12.4+ support)
* `transformers`, `huggingface_hub`
* `openai-whisper`
* `moviepy`, `opencv-python`, `Pillow`
* `deep-translator`, `nltk`, `rake-nltk`

## Usage 🚀

1. Place any video files (`.mp4`, `.mov`, `.mkv`, `.avi`) into the `videos/` folder.
2. Run the extraction script:

```powershell
python videoprofessor/extractMeta.py
```

3. The script will automatically loop through every video, download the necessary AI models to your local cache (if running for the first time), and print `✔ Done: [filename]` as it finishes each video.
4. Your clean metadata will be dumped into `video_metadata.csv`.

## Configuration ⚙️

You can balance speed vs. quality by changing the model sizes directly inside `extractMeta.py`:

```python
# The "Goldilocks" middle-ground setting:
WHISPER_MODEL_SIZE = "small"                # Very accurate for transcription 
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"    # Smart, 1.5 Billion param description generator
```

*For absolute maximum speed, you can downgrade these to `"base"` and `"Qwen/Qwen2.5-0.5B-Instruct"`. For maximum quality (if you have the VRAM), you can upgrade them to `"medium"` and `"Qwen/Qwen2.5-3B-Instruct"`.*
