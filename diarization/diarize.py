"""
Pipeline diaryzacji:
1. Silero VAD  → segmenty mowa/cisza + timestampy
2. pyannote    → przypisanie segmentów do mówców (SPEAKER_00, SPEAKER_01, ...)

Wynik: JSON + CSV per nagranie.

Użycie:
  python diarize.py
"""
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta
from rich.console import Console
from rich.table import Table
from pyannote.audio import Pipeline as DiarizationPipeline

from config import (
    WAV_DIR, JSON_DIR, CSV_DIR, SAMPLE_RATE,
    VAD_THRESHOLD, MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS,
    MIN_SPEAKERS, MAX_SPEAKERS, NUM_SPEAKERS,
    PYANNOTE_MODEL, load_hf_token, ensure_dirs
)

console = Console()


# ─────────────────────────────────────────────
# Silero VAD
# ─────────────────────────────────────────────
_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        _vad_utils = utils
    return _vad_model, _vad_utils


def run_vad(wav_path: Path) -> list[dict]:
    """Zwraca listę segmentów: {start, end, type} (speech/silence)."""
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    audio = read_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    speech_ts = get_speech_timestamps(
        audio, model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS
    )

    segments = []
    prev_end = 0.0

    for ts in speech_ts:
        start_sec = ts["start"] / SAMPLE_RATE
        end_sec = ts["end"] / SAMPLE_RATE

        # Cisza przed mową
        if start_sec - prev_end > 0.1:
            segments.append({
                "start": round(prev_end, 3),
                "end": round(start_sec, 3),
                "type": "silence"
            })

        segments.append({
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "type": "speech"
        })
        prev_end = end_sec

    return segments


# ─────────────────────────────────────────────
# pyannote Speaker Diarization
# ─────────────────────────────────────────────
_diar_pipeline = None


def _load_diarization(hf_token: str):
    global _diar_pipeline
    if _diar_pipeline is None:
        console.print("  [dim]Ładowanie pyannote pipeline...[/dim]")
        _diar_pipeline = DiarizationPipeline.from_pretrained(
            PYANNOTE_MODEL,
            use_auth_token=hf_token
        )
        _diar_pipeline.to(torch.device("cpu"))
    return _diar_pipeline


def run_diarization(wav_path: Path, hf_token: str) -> list[dict]:
    """Zwraca listę: {start, end, speaker}."""
    pipeline = _load_diarization(hf_token)

    # Parametry mówców
    kwargs = {}
    if NUM_SPEAKERS is not None:
        kwargs["num_speakers"] = NUM_SPEAKERS
    else:
        if MIN_SPEAKERS is not None:
            kwargs["min_speakers"] = MIN_SPEAKERS
        if MAX_SPEAKERS is not None:
            kwargs["max_speakers"] = MAX_SPEAKERS

    diarization = pipeline(str(wav_path), **kwargs)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })

    return results


# ─────────────────────────────────────────────
# Merge VAD + Diarization
# ─────────────────────────────────────────────
def merge_results(vad_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
    """Łączy VAD (silence/speech) z diaryzacją (kto mówi)."""
    merged = []

    for seg in vad_segments:
        if seg["type"] == "silence":
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "type": "silence",
                "speaker": None,
                "duration": round(seg["end"] - seg["start"], 3)
            })
        else:
            best_speaker = _find_dominant_speaker(
                seg["start"], seg["end"], diar_segments
            )
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "type": "speech",
                "speaker": best_speaker,
                "duration": round(seg["end"] - seg["start"], 3)
            })

    return merged


def _find_dominant_speaker(start: float, end: float, diar_segments: list[dict]) -> str:
    """Mówca z największym overlapem w danym przedziale."""
    overlaps = {}
    for d in diar_segments:
        overlap_start = max(start, d["start"])
        overlap_end = min(end, d["end"])
        overlap = max(0, overlap_end - overlap_start)
        if overlap > 0:
            overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0) + overlap

    if not overlaps:
        return "UNKNOWN"
    return max(overlaps, key=overlaps.get)


# ─────────────────────────────────────────────
# Formatowanie i eksport
# ─────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_results(filename: str, merged: list[dict]) -> tuple[Path, Path]:
    """Eksportuje wyniki do JSON i CSV."""
    # JSON
    json_path = JSON_DIR / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = CSV_DIR / f"{filename}.csv"
    df = pd.DataFrame(merged)
    df["start_fmt"] = df["start"].apply(fmt_time)
    df["end_fmt"] = df["end"].apply(fmt_time)
    df = df[["start_fmt", "end_fmt", "duration", "type", "speaker"]]
    df.columns = ["start", "end", "duration_sec", "type", "speaker"]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return json_path, csv_path


def print_summary(merged: list[dict], filename: str):
    table = Table(title=f"Podsumowanie: {filename}")
    table.add_column("Mówca", style="cyan")
    table.add_column("Czas mówienia", style="green")
    table.add_column("Segmenty", style="yellow")

    speaker_stats = {}
    for seg in merged:
        if seg["type"] == "speech":
            sp = seg["speaker"]
            if sp not in speaker_stats:
                speaker_stats[sp] = {"time": 0, "count": 0}
            speaker_stats[sp]["time"] += seg["duration"]
            speaker_stats[sp]["count"] += 1

    silence_time = sum(s["duration"] for s in merged if s["type"] == "silence")

    for sp, stats in sorted(speaker_stats.items()):
        table.add_row(sp, fmt_time(stats["time"]), str(stats["count"]))
    table.add_row("CISZA", fmt_time(silence_time), "-", style="dim")

    console.print(table)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def process_file(wav_path: Path, hf_token: str) -> list[dict]:
    filename = wav_path.stem
    console.rule(f"[bold blue]{filename}[/bold blue]")

    console.print("[blue]1/3[/blue] Silero VAD (silence/speech)...")
    vad_segments = run_vad(wav_path)
    speech_count = sum(1 for s in vad_segments if s["type"] == "speech")
    console.print(f"      → {speech_count} segmentów mowy")

    console.print("[blue]2/3[/blue] pyannote diarization (kto mówi)...")
    diar_segments = run_diarization(wav_path, hf_token)
    speakers = set(d["speaker"] for d in diar_segments)
    console.print(f"      → wykryto {len(speakers)} mówców: {speakers}")

    console.print("[blue]3/3[/blue] Merge + export...")
    merged = merge_results(vad_segments, diar_segments)
    json_path, csv_path = export_results(filename, merged)

    print_summary(merged, filename)
    console.print(f"  [green]JSON:[/green] {json_path}")
    console.print(f"  [green]CSV:[/green]  {csv_path}")

    return merged


def process_all():
    ensure_dirs()
    hf_token = load_hf_token()
    wav_files = sorted(WAV_DIR.glob("*.wav"))

    if not wav_files:
        console.print(
            f"[red]Brak plików WAV w {WAV_DIR}/. Uruchom najpierw convert.py[/red]"
        )
        return

    console.print(f"[green]Plików do przetworzenia: {len(wav_files)}[/green]\n")

    for wav in wav_files:
        process_file(wav, hf_token)
        console.print()


if __name__ == "__main__":
    process_all()
