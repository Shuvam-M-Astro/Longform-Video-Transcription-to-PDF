# Video Documentation Builder

Turn long-form videos into a single PDF that contains:
- Full transcript
- (Optional) Extracted key frames and categorized visuals
- (Optional) Extracted code snippets

The tool downloads (or streams) a video from a URL, transcribes the audio, optionally extracts/classifies frames, and assembles a final PDF.

## Methodology
- Acquisition: Retrieve video via standard web client, with cookie-authentication and fallbacks to bypass YouTube bot checks.
- Audio: Extract mono 16 kHz WAV using FFmpeg (from file or from stream).
- Transcription: Use Faster-Whisper for accurate, timestamped speech-to-text.
- Visuals (optional): Extract keyframes via FFmpeg (scene/iframe/interval); OCR with heuristics to separate code, plots, and images.
- Assembly: Build a clean PDF containing transcript, and when enabled, images/plots and extracted code snippets.

## Requirements
- Python 3.10+
- FFmpeg installed and available in PATH
- (Optional) NVIDIA GPU + CUDA for acceleration (PyTorch build)

### Install FFmpeg (Windows)
- Download from: https://www.gyan.dev/ffmpeg/builds/ (full build)
- Add `bin` to PATH, verify with `ffmpeg -version`

## Quickstart (Anaconda Prompt, Windows)
```cmd
cd "C:\path\to\Video documentation"
conda create -n video-doc python=3.10 -y
conda activate video-doc

:: FFmpeg
conda install -c conda-forge ffmpeg -y

:: Base deps
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Install CUDA 11.8 PyTorch (GPU)
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --upgrade --force-reinstall

:: Verify GPU is visible
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: (Optional) quick-check Faster-Whisper can init on GPU
python -c "from faster_whisper import WhisperModel as W; W('tiny', device='cuda', compute_type='float16'); print('faster-whisper GPU OK')"

:: Run (download + full pipeline)
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --language auto
```

## Interactive UI (Streamlit)

If you prefer to review/edit the transcript, pick keyframes/visuals, and export the PDF without using the CLI, launch the UI:

```cmd
conda activate video-doc
streamlit run app.py
```

Features:
- Transcript editor: edit segments, merge/split rows, save changes
- Keyframe controls: choose extraction method and limits; optional contact sheet
- Visual classification: OCR-based grouping into code, plots, images
- Live PDF build: one-click export with minimal/book layouts

Outputs are written to `outputs/run_ui` by default. You can change the output directory in the sidebar.

Notes:
- If you prefer CUDA 12.1, install the matching PyTorch wheels (cu121) from PyTorch.
- Close your browser when using cookie-based extraction.

## Usage
Basic run (download flow):
```cmd
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1"
```

Transcribe-only (skip frames and classification; fastest):
```cmd
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --transcribe-only --language en --beam-size 1 --whisper-model small
```

### Use an existing local video file (no download)
You can process a file you already have on disk with `--video` instead of `--url`.

```cmd
:: Local MP4 (Windows paths need quotes if they include spaces)
python main.py --video "C:\path\to\your\video.mp4" --out ".\outputs\run1"

:: Force language/model (optional)
python main.py --video "C:\path\to\your\video.mp4" --out ".\outputs\run1" --language en --beam-size 1 --whisper-model small

:: Audio-only files also work (mp3/m4a/wav/flac)
python main.py --video ".\inputs\lecture_audio.mp3" --out ".\outputs\run1"
```

Notes:
- `--video` accepts any container/codec FFmpeg can read (tested: mp4, mkv, mov, webm; audio: mp3, m4a, wav, flac).
- You do not need `--skip-download` when using `--video` (no network download happens).
- Output structure is the same, except no `video.mp4` is saved in the run folder.

Reuse existing download:
```cmd
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --skip-download
```

Streaming mode (no full MP4 saved; add cookies if needed):
```cmd
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --streaming --cookies-from-browser chrome --browser-profile Default
```

Keyframe extraction methods (when not using transcribe-only):
- scene-based (default): `--kf-method scene --max-fps 0.2 --min-scene-diff 0.7`
- I-frames (fastest): `--kf-method iframe`
- time interval: `--kf-method interval --kf-interval-sec 10`

Examples:
```cmd
:: I-frames only
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --skip-download --kf-method iframe

:: Every 10 seconds
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --skip-download --kf-method interval --kf-interval-sec 10
```

Bypass YouTube bot checks (optional):
```cmd
:: Chrome cookies
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --cookies-from-browser chrome --browser-profile Default

:: Edge cookies
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --cookies-from-browser edge --browser-profile Default

:: cookies.txt file
python main.py --url "<VIDEO_URL>" --out ".\outputs\run1" --cookies-file ".\cookies.txt"
```

## Output Structure
```
outputs/
  run1/
    video.mp4              # when download mode
    audio.wav
    transcript/
      transcript.txt
      segments.json
    frames/
      keyframes/
        frame_000123.jpg   # when frames enabled
    classified/
      code/                # when frames enabled
      plots/
      images/
    snippets/
      code/
    report.pdf
```

## Troubleshooting
- Stuck on keyframes: switch to `--kf-method iframe` or `--kf-method interval --kf-interval-sec 10`.
- Slow transcription on CPU: use `--whisper-model small` and `--beam-size 1` (or enable GPU).
- CUDA errors: script auto-falls back to CPU; install correct PyTorch CUDA build for GPU.
