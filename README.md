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

## Batch Processing

The tool now supports batch processing of multiple videos with advanced features like parallel processing, progress tracking, and error handling.

### Basic Batch Processing

Process multiple videos from a text file:
```cmd
python batch_process.py --input-file videos.txt --output-dir ./batch_output
```

Process all videos in a directory:
```cmd
python batch_process.py --input-dir ./videos --output-dir ./batch_output --parallel 2
```

Process specific URLs:
```cmd
python batch_process.py --urls "url1,url2,url3" --output-dir ./batch_output
```

### Advanced Batch Processing

Use YAML configuration for complex batch processing:
```cmd
python batch_process_advanced.py --config batch_config.yaml
```

### Batch Processing Features

- **Parallel Processing**: Process multiple videos simultaneously
- **Progress Tracking**: Real-time progress updates and ETA
- **Error Handling**: Retry failed videos with configurable attempts
- **Resume Capability**: Resume interrupted batch processing
- **Priority Support**: Process high-priority videos first
- **State Management**: Save and restore processing state
- **Detailed Logging**: Comprehensive logs and result reports
- **Flexible Input**: Support for files, directories, and URLs

### Creating Video Lists

Create a video list from a directory:
```cmd
python batch_utils.py create-list ./videos ./video_list.txt --recursive
```

Create a video list from URLs:
```cmd
python batch_utils.py create-urls "url1" "url2" "url3" ./url_list.txt
```

### Analyzing Results

View batch processing summary:
```cmd
python batch_utils.py analyze ./batch_output/batch_results.json
```

Export results to CSV:
```cmd
python batch_utils.py analyze ./batch_output/batch_results.json --csv results.csv
```

### Batch Configuration

Create a configuration template:
```cmd
python batch_utils.py template batch_config.yaml
```

Example batch configuration:
```yaml
batch:
  max_parallel: 2
  retry_failed: true
  max_retries: 2
  skip_existing: true
  stop_on_error: false
  timeout_per_video: 3600

processing:
  transcribe_only: false
  language: "auto"
  whisper_model: "medium"
  beam_size: 5
  report_style: "book"

keyframes:
  method: "scene"
  max_fps: 1.0
  max_frames: 0
```

## Search & Indexing

The tool automatically indexes transcripts for cross-language search after processing. You can manage and search indexes using the command-line interface.

### Check Indexing Status

Check if a processed video has been indexed:

```cmd
python search_cli.py status --output-dir ./outputs/run1
```

Or using a job ID:

```cmd
python search_cli.py status --job-id <job-uuid>
```

### Manually Index a Transcript

If indexing failed or you want to re-index:

```cmd
python search_cli.py index --output-dir ./outputs/run1
```

### Search Transcripts

Search across all indexed transcripts:

```cmd
# Basic search
python search_cli.py search --query "machine learning"

# Search with more results and lower score threshold
python search_cli.py search --query "deep learning" --limit 20 --min-score 0.3

# Search with translation to Spanish
python search_cli.py search --query "neural networks" --target-language es

# Export results to JSON
python search_cli.py search --query "AI" --output results.json --format json

# Use keyword search mode
python search_cli.py search --query "python programming" --mode keyword

# Hybrid search (combines semantic and keyword)
python search_cli.py search --query "data science" --mode hybrid
```

### List Indexed Jobs

View all jobs that have been indexed:

```cmd
python search_cli.py list
```

For more details on search functionality, see [CROSS_LANGUAGE_SEARCH_README.md](CROSS_LANGUAGE_SEARCH_README.md).

## Troubleshooting
- Stuck on keyframes: switch to `--kf-method iframe` or `--kf-method interval --kf-interval-sec 10`.
- Slow transcription on CPU: use `--whisper-model small` and `--beam-size 1` (or enable GPU).
- CUDA errors: script auto-falls back to CPU; install correct PyTorch CUDA build for GPU.
- Batch processing issues: check logs in `batch_log.txt` and use `--parallel 1` for debugging.
- Search returns no results: check indexing status with `python search_cli.py status`, try lowering `--min-score`, or use `--mode keyword` for exact matches.
