"""
Transkrypcja audio z przypisaniem tekstu do mówców.

Strategia: transkrybuje cały plik WAV jednym przejściem (lepszy kontekst),
potem mapuje słowa na segmenty diaryzacji po timestampach.

Użycie:
  python transcribe.py                # transkrybuj wszystkie nagrania
  python transcribe.py meeting_001    # transkrybuj jedno nagranie
"""
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from config import (
    WAV_DIR, JSON_DIR, CSV_DIR, TRANSCRIPT_DIR,
    WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE, WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS, WHISPER_MODEL_DIR,
    ensure_dirs
)

console = Console()

# ─────────────────────────────────────────────
# Model faster-whisper (singleton)
# ─────────────────────────────────────────────
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        console.print("  [dim]Ladowanie modelu faster-whisper...[/dim]")
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_MODEL_DIR,
            cpu_threads=WHISPER_CPU_THREADS,
        )
    return _whisper_model


# ─────────────────────────────────────────────
# Transkrypcja
# ─────────────────────────────────────────────
def transcribe_wav(wav_path: Path) -> list[dict]:
    """
    Transkrybuje caly plik WAV.
    Zwraca liste slow: [{"word": str, "start": float, "end": float}, ...]
    """
    model = _load_whisper()

    segments_iter, info = model.transcribe(
        str(wav_path),
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
    )

    console.print(
        f"  [dim]Wykryty jezyk: {info.language} "
        f"(prawdopodobienstwo: {info.language_probability:.1%})[/dim]"
    )

    words = []
    for segment in segments_iter:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                })

    return words


# ─────────────────────────────────────────────
# Mapowanie slow na segmenty diaryzacji
# ─────────────────────────────────────────────
def map_words_to_segments(
    words: list[dict],
    segments: list[dict]
) -> list[dict]:
    """
    Przypisuje slowa z Whisper do segmentow diaryzacji po timestampach.
    Dodaje pole "text" do kazdego segmentu speech.
    """
    # Indeks segmentow speech
    speech_segments = [
        (i, seg) for i, seg in enumerate(segments)
        if seg["type"] == "speech"
    ]

    # Inicjalizuj tekst
    seg_words: dict[int, list[str]] = {i: [] for i, _ in speech_segments}

    for w in words:
        word_mid = (w["start"] + w["end"]) / 2
        best_idx = _find_closest_segment(word_mid, speech_segments)
        if best_idx is not None:
            seg_words[best_idx].append(w["word"])

    # Dodaj tekst do segmentow
    for i, seg in enumerate(segments):
        if seg["type"] == "speech" and i in seg_words:
            segments[i]["text"] = "".join(seg_words[i]).strip()
        elif seg["type"] == "speech":
            segments[i]["text"] = ""

    return segments


def _find_closest_segment(
    timestamp: float,
    speech_segments: list[tuple[int, dict]]
) -> int | None:
    """Znajduje segment speech zawierajacy timestamp lub najblizszy."""
    if not speech_segments:
        return None

    # Szukaj segmentu zawierajacego timestamp
    for idx, seg in speech_segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return idx

    # Fallback: najblizszy segment
    best_idx = None
    best_dist = float("inf")
    for idx, seg in speech_segments:
        mid = (seg["start"] + seg["end"]) / 2
        dist = abs(timestamp - mid)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    # Nie przypisuj slow zbyt daleko od segmentu (>5s)
    if best_dist > 5.0:
        return None

    return best_idx


# ─────────────────────────────────────────────
# Eksport
# ─────────────────────────────────────────────
def export_transcript(filename: str, segments: list[dict]):
    """Eksportuje transkrypcje do pliku TXT + aktualizuje JSON i CSV."""
    # Czytelna transkrypcja TXT
    txt_path = TRANSCRIPT_DIR / f"{filename}.txt"
    lines = []
    current_speaker = None

    for seg in segments:
        if seg["type"] != "speech":
            continue
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker = seg.get("matched_speaker") or seg.get("speaker") or "?"
        start_fmt = _fmt_time(seg["start"])

        if speaker != current_speaker:
            if lines:
                lines.append("")
            lines.append(f"[{start_fmt}] {speaker}:")
            current_speaker = speaker

        lines.append(f"  {text}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Aktualizuj JSON z polem "text"
    json_path = JSON_DIR / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # Aktualizuj CSV z kolumna "text"
    csv_path = CSV_DIR / f"{filename}.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # Dodaj kolumne text z segmentow (tylko speech)
        text_map = {}
        for seg in segments:
            if seg["type"] == "speech" and seg.get("text"):
                key = (round(seg["start"], 3), round(seg["end"], 3))
                text_map[key] = seg["text"]

        def _get_text(row):
            try:
                parts = row["start"].split(":")
                h, m = int(parts[0]), int(parts[1])
                s_ms = parts[2].split(".")
                s, ms = int(s_ms[0]), int(s_ms[1])
                start_sec = h * 3600 + m * 60 + s + ms / 1000
            except (ValueError, IndexError, AttributeError):
                return ""
            # Szukaj najblizszego segmentu
            best_text = ""
            best_dist = float("inf")
            for (seg_start, _), text in text_map.items():
                dist = abs(start_sec - seg_start)
                if dist < best_dist:
                    best_dist = dist
                    best_text = text
            return best_text if best_dist < 0.5 else ""

        df["text"] = df.apply(_get_text, axis=1)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return txt_path


def _fmt_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ─────────────────────────────────────────────
# Podsumowanie
# ─────────────────────────────────────────────
def print_transcript_summary(segments: list[dict], filename: str):
    table = Table(title=f"Transkrypcja: {filename}")
    table.add_column("Mowca", style="cyan")
    table.add_column("Segmenty z tekstem", style="green")
    table.add_column("Laczna dl. tekstu", style="yellow")

    speaker_stats: dict[str, dict] = {}
    for seg in segments:
        if seg["type"] != "speech":
            continue
        sp = seg.get("matched_speaker") or seg.get("speaker") or "?"
        text = seg.get("text", "")
        if sp not in speaker_stats:
            speaker_stats[sp] = {"count": 0, "chars": 0}
        if text.strip():
            speaker_stats[sp]["count"] += 1
            speaker_stats[sp]["chars"] += len(text)

    for sp, stats in sorted(speaker_stats.items()):
        table.add_row(sp, str(stats["count"]), f"{stats['chars']} znakow")

    console.print(table)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def transcribe_file(wav_name: str) -> list[dict]:
    """Transkrybuje jedno nagranie i mapuje na segmenty diaryzacji."""
    wav_path = WAV_DIR / f"{wav_name}.wav"
    json_path = JSON_DIR / f"{wav_name}.json"

    if not wav_path.exists():
        console.print(f"[red]Brak {wav_path}[/red]")
        return []
    if not json_path.exists():
        console.print(f"[red]Brak {json_path}. Uruchom najpierw diarize.py[/red]")
        return []

    console.rule(f"[bold blue]Transkrypcja: {wav_name}[/bold blue]")

    # Wczytaj segmenty diaryzacji
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Transkrybuj
    console.print("[blue]1/2[/blue] Whisper: transkrypcja calego pliku...")
    words = transcribe_wav(wav_path)
    console.print(f"      -> {len(words)} slow")

    # Mapuj slowa na mowcow
    console.print("[blue]2/2[/blue] Mapowanie slow na segmenty diaryzacji...")
    segments = map_words_to_segments(words, segments)

    # Eksport
    txt_path = export_transcript(wav_name, segments)
    print_transcript_summary(segments, wav_name)

    console.print(f"  [green]TXT:[/green] {txt_path}")

    return segments


def transcribe_all():
    """Transkrybuje wszystkie nagrania z wynikami diaryzacji."""
    ensure_dirs()
    wav_files = sorted(WAV_DIR.glob("*.wav"))

    if not wav_files:
        console.print(f"[red]Brak plikow WAV w {WAV_DIR}/[/red]")
        return

    count = 0
    for wav in wav_files:
        wav_name = wav.stem
        json_path = JSON_DIR / f"{wav_name}.json"
        if json_path.exists():
            transcribe_file(wav_name)
            count += 1
            console.print()

    if count == 0:
        console.print("[yellow]Brak wynikow diaryzacji. Uruchom najpierw diarize.py[/yellow]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1])
    else:
        transcribe_all()
