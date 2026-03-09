"""
Konwersja MP4 → WAV 16kHz mono (optymalny format dla diaryzacji).

Użycie:
  python convert.py
  → przetwarza wszystkie MP4 z input/ do output/wav/
"""
import subprocess
from pathlib import Path
from rich.console import Console

from config import INPUT_DIR, WAV_DIR, SAMPLE_RATE, CHANNELS, ensure_dirs

console = Console()


def convert_file(mp4_path: Path) -> Path:
    """Konwertuje pojedynczy MP4 → WAV."""
    wav_path = WAV_DIR / mp4_path.with_suffix(".wav").name

    if wav_path.exists():
        console.print(f"  [yellow]SKIP[/yellow] {mp4_path.name} (WAV istnieje)")
        return wav_path

    console.print(f"  [blue]Konwersja[/blue] {mp4_path.name}...")
    cmd = [
        "ffmpeg", "-i", str(mp4_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-vn",                    # bez wideo
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-y",                     # nadpisz bez pytania
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"  [red]BŁĄD[/red] {mp4_path.name}: {result.stderr[-200:]}")
        return None

    console.print(f"  [green]OK[/green] → {wav_path.name}")
    return wav_path


def convert_all() -> list[Path]:
    """Konwertuje wszystkie MP4 z input/."""
    ensure_dirs()
    mp4_files = sorted(INPUT_DIR.glob("*.mp4"))

    if not mp4_files:
        console.print(f"[red]Brak plików MP4 w {INPUT_DIR}/[/red]")
        return []

    console.print(f"[green]Znaleziono {len(mp4_files)} plików MP4[/green]")
    wav_files = []

    for mp4 in mp4_files:
        wav = convert_file(mp4)
        if wav:
            wav_files.append(wav)

    console.print(f"[green]Skonwertowano: {len(wav_files)}/{len(mp4_files)}[/green]")
    return wav_files


if __name__ == "__main__":
    convert_all()
