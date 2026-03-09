"""
Ręczne mapowanie etykiet na prawdziwe imiona (fallback).
Również: rozwiązywanie AMBIGUOUS matchów z match_speakers.py.

Użycie:
  1. Przejrzyj CSV → kolumna matched_speaker
  2. Jeśli ⚠ AMBIGUOUS → odsłuchaj segmenty i zdecyduj
  3. Wpisz mapowanie w SPEAKER_MAP poniżej
  4. python label_speakers.py
"""
import json
import pandas as pd
from pathlib import Path
from rich.console import Console

from config import JSON_DIR, CSV_DIR

console = Console()

# ============================================================
# EDYTUJ TUTAJ: mapowanie per nagranie
# ============================================================
SPEAKER_MAP = {
    # "meeting_001": {
    #     "SPEAKER_00": "Anna Kowalska",
    #     "SPEAKER_01": "Marek Nowak",
    # },
    # "meeting_002": {
    #     "SPEAKER_03": "Anna Kowalska",  # pyannote numeruje inaczej per nagranie
    # },
}

# Mapowanie globalne (fallback — jeśli nie ma per-plik):
GLOBAL_MAP = {
    # "SPEAKER_00": "Hubert",
}
# ============================================================


def relabel():
    count = 0
    for json_path in sorted(JSON_DIR.glob("*.json")):
        name = json_path.stem
        mapping = SPEAKER_MAP.get(name, GLOBAL_MAP)
        if not mapping:
            continue

        # JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for seg in data:
            for field in ["speaker", "matched_speaker"]:
                if seg.get(field) in mapping:
                    seg[field] = mapping[seg[field]]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # CSV
        csv_path = CSV_DIR / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            for col in ["speaker", "matched_speaker"]:
                if col in df.columns:
                    df[col] = df[col].map(lambda x: mapping.get(x, x))
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        console.print(f"[green]✓[/green] Relabeled: {name} ({len(mapping)} mówców)")
        count += 1

    if count == 0:
        console.print("[yellow]Brak mapowań do zastosowania.[/yellow]")
        console.print("Edytuj SPEAKER_MAP w label_speakers.py")


if __name__ == "__main__":
    relabel()
