# Voice Diarization Pipeline

## Project Overview
Offline pipeline for processing MP4 video conference recordings:
1. MP4 → WAV conversion (FFmpeg, 16kHz mono)
2. VAD — Voice Activity Detection (Silero VAD): silence vs speech + timestamps
3. Speaker Diarization (pyannote.audio 3.1): who speaks when
4. Transcription (faster-whisper, large-v3-turbo): what is being said
5. Cross-recording Speaker Matching (SpeechBrain ECAPA-TDNN): consistent speaker identity across recordings

## Key Constraints
- **~7 regular speakers** (enrolled in speaker database with multi-embedding profiles)
- **7-8 unique guests per recording** (clustered ad-hoc, labeled GUEST_XXX)
- **2 speakers have very similar voices** → multi-embedding matching + AMBIGUITY_MARGIN detection + prosodic features as tiebreaker
- **Runs offline** on Dell Intel Ultra 7, CPU-only (no GPU required)

## Architecture

```
input/*.mp4
    ↓ convert.py (FFmpeg)
output/wav/*.wav
    ↓ diarize.py (Silero VAD + pyannote)
output/json/*.json + output/csv/*.csv
    ↓ transcribe.py (faster-whisper)
output/json/*.json + output/csv/*.csv (with text column) + output/transcript/*.txt
    ↓ match_speakers.py (ECAPA-TDNN embeddings vs speaker_db)
output/json/*.json + output/csv/*.csv (with matched_speaker column)
```

## File Structure
```
convert.py          — MP4 → WAV batch conversion
diarize.py          — VAD + speaker diarization pipeline
transcribe.py       — Whisper transcription + word-to-speaker mapping
speaker_db.py       — Speaker database management (multi-embedding profiles)
enroll_speaker.py   — Enroll known speakers (from audio samples or diarization results)
match_speakers.py   — Cross-recording speaker matching (enrolled vs guests)
label_speakers.py   — Manual fallback: relabel speakers by hand
run_all.py          — Master script: convert → diarize → transcribe → match
save_models.py      — One-time model download for offline use
config.py           — All configurable parameters in one place
requirements.txt    — Python dependencies
.env                — HuggingFace token (HF_TOKEN=hf_...)
```

## Setup
```powershell
uv sync                          # tworzy .venv + instaluje wszystko
# Edit .env with HuggingFace token
uv run save_models.py            # one-time download
```

## Usage
```powershell
# Full pipeline:
uv run run_all.py

# Individual steps:
uv run convert.py
uv run diarize.py
uv run transcribe.py
uv run enroll_speaker.py
uv run match_speakers.py
uv run label_speakers.py
```

## Tech Stack
- **PyTorch** (CPU-only) — tensor backend
- **Silero VAD** — voice activity detection (~2MB model)
- **pyannote.audio 3.1** — speaker diarization (~500MB models, HuggingFace token required)
- **SpeechBrain ECAPA-TDNN** — speaker embeddings for cross-recording matching (~25MB)
- **faster-whisper** (CTranslate2) — speech-to-text transcription (~1.5GB model, INT8 quantized)
- **scikit-learn** — agglomerative clustering for guest speakers
- **FFmpeg** — audio extraction from MP4

## Key Design Decisions
- WAV 16kHz mono (not MP3) — lossless for analysis
- Multi-embedding profiles (not single centroid) — handles voice variability and similar voices
- MAX similarity matching (not mean) — catches best discriminating moments
- AMBIGUITY_MARGIN flag — refuses to guess on similar voices, asks for human review
- Self-learning — every HIGH confidence match enriches the speaker profile
- Two-tier logic — enrolled (known) vs guest (unknown) speakers
- Full-file transcription with word-level timestamp mapping (not per-segment) — better Whisper accuracy
