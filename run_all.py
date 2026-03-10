"""
Master script: convert → diarize → match (jeśli baza enrolled istnieje).

Użycie:
  python run_all.py
"""
from rich.console import Console
from config import ensure_dirs

console = Console()


def main():
    ensure_dirs()

    # KROK 1: Konwersja
    console.rule("[bold green]KROK 1: Konwersja MP4 → WAV[/bold green]")
    from convert import convert_all
    convert_all()

    # KROK 2: Diaryzacja
    console.rule("[bold green]KROK 2: Diaryzacja (VAD + Speaker ID)[/bold green]")
    from diarize import process_all
    process_all()

    # KROK 3: Transkrypcja
    console.rule("[bold green]KROK 3: Transkrypcja (Whisper)[/bold green]")
    from transcribe import transcribe_all
    transcribe_all()

    # KROK 4: Matching (opcjonalny — wymaga enrolled w bazie)
    from speaker_db import SpeakerDatabase
    db = SpeakerDatabase()
    enrolled = db.get_all_enrolled()

    if enrolled:
        console.rule("[bold green]KROK 4: Cross-recording matching[/bold green]")
        console.print(f"Enrolled w bazie: {[p.name for p in enrolled]}")
        from match_speakers import match_all
        match_all(update_profiles=True)
    else:
        console.print("\n[yellow]Brak enrolled mówców w bazie.[/yellow]")
        console.print("Aby włączyć cross-recording matching:")
        console.print("  1. Przejrzyj wyniki diaryzacji w output/csv/")
        console.print("  2. Uruchom: [bold]python enroll_speaker.py[/bold]")
        console.print("  3. Potem:   [bold]python match_speakers.py[/bold]")

    console.rule("[bold green]GOTOWE[/bold green]")
    console.print("Wyniki w folderach: output/json/, output/csv/ i output/transcript/")


if __name__ == "__main__":
    main()
