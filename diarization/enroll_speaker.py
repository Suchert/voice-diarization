"""
Enrollment: budowanie profili znanych mówców.

DWA TRYBY:
──────────
1. sample      — z próbek audio (enrollment_samples/*.wav, min 30s per osoba)
2. diarization — z wyników diaryzacji (wskaż nagranie + SPEAKER_XX)

WAŻNE DLA PODOBNYCH GŁOSÓW:
────────────────────────────
Im więcej embeddingów per osoba, tym lepsze rozróżnienie.
Idealnie: enrollment z 3-5 różnych nagrań per osoba.

Użycie:
  python enroll_speaker.py
"""
import win_compat  # noqa: F401 — monkey-patches os.symlink on Windows

import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from speechbrain.inference.speaker import EncoderClassifier

from config import (
    ENROLLMENT_DIR, WAV_DIR, JSON_DIR,
    SAMPLE_RATE, MIN_SEGMENT_DURATION_S,
    ECAPA_MODEL, ECAPA_SAVEDIR, ensure_dirs
)
from speaker_db import SpeakerDatabase

console = Console()

# ─────────────────────────────────────────────
# Model ECAPA-TDNN (singleton)
# ─────────────────────────────────────────────
_classifier = None


def get_classifier() -> EncoderClassifier:
    global _classifier
    if _classifier is None:
        console.print("[dim]Ładowanie modelu ECAPA-TDNN...[/dim]")
        _classifier = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL,
            savedir=ECAPA_SAVEDIR,
            run_opts={"device": "cpu"},
        )
    return _classifier


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────
def extract_embedding(wav_path: Path) -> np.ndarray:
    """Wyciąga embedding 192-dim z pliku WAV."""
    classifier = get_classifier()
    signal, sr = torchaudio.load(str(wav_path))

    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().cpu().numpy()


def extract_embedding_from_segments(
    wav_path: Path,
    segments: list[dict]
) -> list[np.ndarray]:
    """Wyciąga per-segment embeddingi (multi-profile)."""
    signal, sr = torchaudio.load(str(wav_path))
    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    classifier = get_classifier()
    all_embeddings = []

    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = signal[:, start_sample:end_sample]

        # Min duration dla sensownego embeddingu
        if chunk.shape[1] >= int(sr * MIN_SEGMENT_DURATION_S):
            emb = classifier.encode_batch(chunk)
            all_embeddings.append(emb.squeeze().cpu().numpy())

    return all_embeddings


def compute_prosodic_features(wav_path: Path, segments: list[dict]) -> dict:
    """Dodatkowe cechy prozodyczne — tiebreaker dla podobnych głosów."""
    signal, sr = torchaudio.load(str(wav_path))
    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    energies = []
    speech_durations = []

    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = signal[0, start_sample:end_sample].numpy()

        if len(chunk) < sr * 0.5:
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)
        speech_durations.append(seg["end"] - seg["start"])

    return {
        "energy_mean": float(np.mean(energies)) if energies else 0,
        "energy_std": float(np.std(energies)) if energies else 0,
        "avg_segment_duration": float(np.mean(speech_durations)) if speech_durations else 0,
        "num_segments": len(speech_durations),
        "total_speech_seconds": float(sum(speech_durations))
    }


# ─────────────────────────────────────────────
# Tryb 1: Z próbek audio
# ─────────────────────────────────────────────
def enroll_from_samples():
    db = SpeakerDatabase()
    samples = sorted(ENROLLMENT_DIR.glob("*.wav"))

    if not samples:
        console.print(f"[red]Brak plików WAV w {ENROLLMENT_DIR}/[/red]")
        console.print("Wrzuć pliki WAV nazwane imieniem, np.:")
        console.print("  enrollment_samples/Anna_Kowalska.wav")
        return

    for sample in samples:
        name = sample.stem.replace("_", " ")
        existing = db.get_profile_by_name(name)

        if existing:
            console.print(f"[yellow]{name}[/yellow] — profil istnieje, dodaję embedding...")
            speaker_id = existing.speaker_id
        else:
            console.print(f"[green]{name}[/green] — nowy profil...")
            profile = db.add_profile(name, is_enrolled=True)
            speaker_id = profile.speaker_id

        embedding = extract_embedding(sample)
        db.add_embedding(speaker_id, embedding,
                         recording_name=f"enrollment:{sample.name}")
        console.print(f"  [green]✓[/green] Embedding {embedding.shape[0]}-dim")

    db.summary()


# ─────────────────────────────────────────────
# Tryb 2: Z diaryzacji
# ─────────────────────────────────────────────
def enroll_from_diarization(wav_name: str, diar_speaker: str, real_name: str):
    """
    Enrollment z wyników diaryzacji.

    Args:
        wav_name: nazwa pliku (bez rozszerzenia), np. "meeting_001"
        diar_speaker: etykieta z diaryzacji, np. "SPEAKER_02"
        real_name: prawdziwe imię, np. "Anna Kowalska"
    """
    db = SpeakerDatabase()
    json_path = JSON_DIR / f"{wav_name}.json"

    if not json_path.exists():
        console.print(f"[red]Nie znaleziono {json_path}. Uruchom najpierw diarize.py[/red]")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    speaker_segments = [
        s for s in segments
        if s.get("speaker") == diar_speaker and s["type"] == "speech"
    ]

    if not speaker_segments:
        console.print(f"[red]Brak segmentów dla {diar_speaker} w {wav_name}[/red]")
        available = set(s.get("speaker") for s in segments if s["type"] == "speech")
        console.print(f"  Dostępni mówcy: {sorted(available)}")
        return

    total_speech = sum(s["duration"] for s in speaker_segments)
    console.print(
        f"[blue]{real_name}[/blue] ← {diar_speaker} — "
        f"{len(speaker_segments)} segmentów, {total_speech:.1f}s mowy"
    )

    wav_path = WAV_DIR / f"{wav_name}.wav"
    if not wav_path.exists():
        console.print(f"[red]Brak {wav_path}[/red]")
        return

    embeddings = extract_embedding_from_segments(wav_path, speaker_segments)

    existing = db.get_profile_by_name(real_name)
    if existing:
        speaker_id = existing.speaker_id
        console.print(f"  Profil istnieje — dodaję {len(embeddings)} embeddingów...")
    else:
        profile = db.add_profile(real_name, is_enrolled=True)
        speaker_id = profile.speaker_id
        console.print(f"  Nowy profil — dodaję {len(embeddings)} embeddingów...")

    for emb in embeddings:
        db.add_embedding(
            speaker_id, emb,
            recording_name=wav_name,
            speech_seconds=total_speech / max(len(embeddings), 1)
        )

    # Cechy prozodyczne
    prosodic = compute_prosodic_features(wav_path, speaker_segments)
    db.profiles[speaker_id].prosodic_features = prosodic
    db.save()

    console.print(f"  [green]✓[/green] {len(embeddings)} embeddingów + cechy prozodyczne")
    db.summary()


# ─────────────────────────────────────────────
# Interaktywny enrollment
# ─────────────────────────────────────────────
def interactive_enroll():
    ensure_dirs()
    console.rule("[bold blue]Enrollment mówców[/bold blue]")
    mode = Prompt.ask("Tryb", choices=["sample", "diarization"])

    if mode == "sample":
        enroll_from_samples()
    else:
        wav_name = Prompt.ask("Nazwa pliku WAV (bez rozszerzenia)")
        diar_speaker = Prompt.ask("Etykieta z diaryzacji (np. SPEAKER_00)")
        real_name = Prompt.ask("Prawdziwe imię i nazwisko")
        enroll_from_diarization(wav_name, diar_speaker, real_name)


if __name__ == "__main__":
    interactive_enroll()
