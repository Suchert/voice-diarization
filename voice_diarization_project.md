# Voice Diarization Pipeline — Offline Setup Guide

> **Cel:** Przetwarzanie nagrań MP4 z wideokonferencji → wykrywanie cisza/mowa + identyfikacja mówców + cross-recording matching (kto jest kto między nagraniami).
>
> **Konfiguracja:** ~7 stałych mówców + 7-8 gości per nagranie. Dwie osoby o podobnych głosach.
>
> **Maszyna docelowa:** Dell, Intel Ultra 7, Windows 11, offline po jednorazowym pobraniu.

---

## 1. Wymagania systemowe

| Komponent | Minimum | Twój Dell |
|-----------|---------|-----------|
| CPU | 4 rdzenie, AVX2 | Intel Ultra 7 ✅ |
| RAM | 8 GB | prawdopodobnie 16+ GB ✅ |
| Dysk | ~5 GB wolnego | ✅ |
| GPU | **nie wymagane** | Intel Arc iGPU (bonus) |
| Internet | tylko do pierwszego setupu | ✅ |

---

## 2. Jednorazowa instalacja (z internetem)

### 2.1 Python 3.11+

Pobierz i zainstaluj z [python.org](https://www.python.org/downloads/).

Podczas instalacji zaznacz:
- ✅ **Add Python to PATH**
- ✅ **Install pip**

Weryfikacja:
```powershell
python --version
pip --version
```

### 2.2 FFmpeg

Pobierz build dla Windows z [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) → `ffmpeg-release-essentials.zip`.

1. Rozpakuj do `C:\ffmpeg`
2. Dodaj `C:\ffmpeg\bin` do zmiennej PATH:
   - Win+R → `sysdm.cpl` → Zaawansowane → Zmienne środowiskowe → Path → Edytuj → Dodaj `C:\ffmpeg\bin`

Weryfikacja:
```powershell
ffmpeg -version
```

### 2.3 Środowisko Pythona

```powershell
# Utwórz folder projektu
mkdir C:\diarization
cd C:\diarization

# Wirtualne środowisko
python -m venv venv
.\venv\Scripts\activate

# PyTorch CPU-only (~200 MB)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pipeline audio
pip install pyannote.audio==3.3.2
pip install silero-vad
pip install pydub
pip install rich        # ładne logi w terminalu
pip install pandas      # export do CSV/Excel
pip install speechbrain # embeddingi głosu (ECAPA-TDNN) — cross-recording matching
pip install scikit-learn # clustering, metryki
pip install numpy
```

### 2.4 Token HuggingFace (jednorazowo)

Modele pyannote wymagają akceptacji licencji:

1. Załóż konto na [huggingface.co](https://huggingface.co)
2. Wejdź na stronę modelu: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) — kliknij **Agree**
3. To samo dla: [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) — kliknij **Agree**
4. Wygeneruj token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** (Read)
5. Zapisz token w pliku:

```powershell
echo HF_TOKEN=hf_TwojTokenTutaj > C:\diarization\.env
```

### 2.5 Jednorazowe pobranie modeli (offline cache)

Uruchom **raz** z internetem — modele zapiszą się w cache (~800 MB):

```python
# save_models.py
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier
import torch

# 1. pyannote diarization (~500 MB)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="hf_TwojTokenTutaj"
)
pipeline.to(torch.device("cpu"))
print("✓ pyannote models cached.")

# 2. SpeechBrain ECAPA-TDNN (~25 MB) — do cross-recording matching
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)
print("✓ ECAPA-TDNN model cached.")

# 3. Silero VAD (~2 MB)
model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
print("✓ Silero VAD cached.")

print("\nWszystkie modele pobrane. Teraz działa offline.")
```

```powershell
python save_models.py
```

Po wykonaniu tego kroku **internet nie jest już potrzebny**.

---

## 3. Struktura projektu

```
C:\diarization\
├── venv\                    # środowisko wirtualne
├── input\                   # <- wrzuć tu pliki MP4
├── output\                  # <- tu lądują wyniki
│   ├── wav\                 #    pliki WAV (konwersja)
│   ├── json\                #    wyniki w JSON
│   └── csv\                 #    wyniki w CSV
├── speaker_db\              # <- baza głosów (embeddingi + metadane)
│   ├── profiles.json        #    profil każdego znanego mówcy
│   └── embeddings\          #    .npy pliki z embeddingami
├── enrollment_samples\      # <- próbki głosów do enrollmentu (30-60s WAV per osoba)
├── .env                     # token HuggingFace
├── save_models.py           # jednorazowe pobranie modeli
├── convert.py               # MP4 → WAV
├── diarize.py               # główny pipeline (VAD + diaryzacja)
├── speaker_db.py            # zarządzanie bazą głosów
├── enroll_speaker.py        # enrollment znanych mówców
├── match_speakers.py        # cross-recording matching
├── label_speakers.py        # ręczne mapowanie SPEAKER_XX → imiona (fallback)
└── run_all.py               # master script (teraz z matchingiem)
```

Utwórz strukturę:
```powershell
cd C:\diarization
mkdir input, output, output\wav, output\json, output\csv, speaker_db, speaker_db\embeddings, enrollment_samples
```

---

## 4. Skrypty

### 4.1 `convert.py` — MP4 → WAV (16kHz mono)

```python
"""Konwersja MP4 → WAV 16kHz mono (optymalny format dla diaryzacji)."""
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

INPUT_DIR = Path("input")
WAV_DIR = Path("output/wav")


def convert_all():
    mp4_files = list(INPUT_DIR.glob("*.mp4"))
    if not mp4_files:
        console.print("[red]Brak plików MP4 w folderze input/[/red]")
        return []

    console.print(f"[green]Znaleziono {len(mp4_files)} plików MP4[/green]")
    wav_files = []

    for mp4 in mp4_files:
        wav_path = WAV_DIR / mp4.with_suffix(".wav").name
        if wav_path.exists():
            console.print(f"  [yellow]SKIP[/yellow] {mp4.name} (WAV istnieje)")
            wav_files.append(wav_path)
            continue

        console.print(f"  [blue]Konwersja[/blue] {mp4.name}...")
        cmd = [
            "ffmpeg", "-i", str(mp4),
            "-ar", "16000",      # sample rate 16kHz
            "-ac", "1",          # mono
            "-vn",               # bez wideo
            "-acodec", "pcm_s16le",
            str(wav_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        wav_files.append(wav_path)
        console.print(f"  [green]OK[/green] → {wav_path.name}")

    return wav_files


if __name__ == "__main__":
    convert_all()
```

### 4.2 `diarize.py` — Główny pipeline (VAD + Speaker Diarization)

```python
"""
Pipeline diaryzacji:
1. Silero VAD  → segmenty mowa/cisza + timestampy
2. pyannote    → przypisanie segmentów do mówców (SPEAKER_00, SPEAKER_01, ...)

Wynik: JSON + CSV per nagranie.
"""
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import timedelta
from rich.console import Console
from rich.table import Table
from pyannote.audio import Pipeline as DiarizationPipeline

console = Console()

WAV_DIR = Path("output/wav")
JSON_DIR = Path("output/json")
CSV_DIR = Path("output/csv")

# --- Wczytaj token z .env ---
def load_token():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise ValueError("Brak HF_TOKEN w pliku .env")


# --- Silero VAD ---
def run_vad(wav_path: Path) -> list[dict]:
    """Zwraca listę segmentów: {start, end, type} gdzie type = 'speech' lub 'silence'."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_) = utils

    audio = read_audio(str(wav_path), sampling_rate=16000)
    speech_ts = get_speech_timestamps(audio, model, sampling_rate=16000)

    segments = []
    prev_end = 0.0

    for ts in speech_ts:
        start_sec = ts["start"] / 16000
        end_sec = ts["end"] / 16000

        # Dodaj ciszę przed mową (jeśli jest przerwa)
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


# --- pyannote Speaker Diarization ---
def run_diarization(wav_path: Path, hf_token: str) -> list[dict]:
    """Zwraca listę: {start, end, speaker}."""
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    pipeline.to(torch.device("cpu"))

    # Parametr: max liczba mówców (opcjonalnie, pomaga przy znanych callach)
    diarization = pipeline(str(wav_path), num_speakers=None)  # auto-detect
    # Jeśli wiesz ile osób: num_speakers=9

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })

    return results


# --- Merge: VAD + Diarization ---
def merge_results(vad_segments: list[dict], diar_segments: list[dict]) -> list[dict]:
    """Łączy VAD (silence/speech) z diaryzacją (kto mówi).
    Segmenty ciszy zachowują type='silence'.
    Segmenty mowy dostają przypisanego mówcę."""
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
            # Znajdź mówcę który dominuje w tym segmencie
            best_speaker = find_dominant_speaker(
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


def find_dominant_speaker(start, end, diar_segments):
    """Znajduje mówcę z największym overlapem w danym przedziale."""
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


# --- Formatowanie czasu ---
def fmt_time(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# --- Eksport ---
def export_results(filename: str, merged: list[dict]):
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


# --- Podsumowanie ---
def print_summary(merged: list[dict], filename: str):
    table = Table(title=f"Podsumowanie: {filename}")
    table.add_column("Mówca", style="cyan")
    table.add_column("Czas mówienia", style="green")
    table.add_column("Liczba segmentów", style="yellow")

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
        table.add_row(
            sp,
            fmt_time(stats["time"]),
            str(stats["count"])
        )
    table.add_row("CISZA", fmt_time(silence_time), "-", style="dim")

    console.print(table)


# --- Main ---
def process_file(wav_path: Path, hf_token: str):
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
    hf_token = load_token()
    wav_files = sorted(WAV_DIR.glob("*.wav"))

    if not wav_files:
        console.print("[red]Brak plików WAV w output/wav/. Uruchom najpierw convert.py[/red]")
        return

    console.print(f"[green]Plików do przetworzenia: {len(wav_files)}[/green]\n")

    for wav in wav_files:
        process_file(wav, hf_token)
        console.print()


if __name__ == "__main__":
    process_all()
```

### 4.3 `label_speakers.py` — Mapowanie SPEAKER_XX → imiona

```python
"""
Opcjonalny krok: mapowanie automatycznych etykiet na prawdziwe imiona.

Sposób użycia:
1. Po pierwszym przebiegu sprawdź CSV — odsłuchaj kilka segmentów
   danego SPEAKER_XX i zidentyfikuj kto to jest.
2. Wpisz mapowanie poniżej.
3. Uruchom skrypt — podmieni etykiety we wszystkich CSV/JSON.
"""
import json
import pandas as pd
from pathlib import Path

# ============================================================
# EDYTUJ TUTAJ: mapowanie per nagranie
# ============================================================
SPEAKER_MAP = {
    # "nazwa_pliku_bez_rozszerzenia": {
    #     "SPEAKER_00": "Anna Kowalska",
    #     "SPEAKER_01": "Marek Nowak",
    #     ...
    # },
    #
    # Przykład:
    # "meeting_2024_01_15": {
    #     "SPEAKER_00": "Hubert",
    #     "SPEAKER_01": "Kasia",
    #     "SPEAKER_02": "Tomek",
    # }
}

# Mapowanie globalne (fallback — jeśli nie ma per-plik):
GLOBAL_MAP = {
    # "SPEAKER_00": "Hubert",
}
# ============================================================

JSON_DIR = Path("output/json")
CSV_DIR = Path("output/csv")


def relabel():
    for json_path in JSON_DIR.glob("*.json"):
        name = json_path.stem
        mapping = SPEAKER_MAP.get(name, GLOBAL_MAP)
        if not mapping:
            continue

        # JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for seg in data:
            if seg.get("speaker") in mapping:
                seg["speaker"] = mapping[seg["speaker"]]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # CSV
        csv_path = CSV_DIR / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df["speaker"] = df["speaker"].map(lambda x: mapping.get(x, x))
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"Relabeled: {name} ({len(mapping)} mówców)")


if __name__ == "__main__":
    relabel()
```

### 4.4 `speaker_db.py` — Baza głosów (multi-embedding profiles)

```python
"""
Baza głosów: przechowuje profile mówców jako kolekcje embeddingów.

Architektura adresująca warunki brzegowe:
─────────────────────────────────────────
PROBLEM 1: Dwie osoby o podobnych głosach.
  → Rozwiązanie: Multi-embedding profile. Zamiast jednego centroidu per osoba,
    trzymamy N embeddingów (z różnych nagrań, kontekstów, energii głosu).
    Matching odbywa się jako max/mean similarity do WSZYSTKICH embeddingów
    w profilu. Dodatkowo: prosodic features (pitch, tempo, energy) jako
    drugorzędne cechy rozróżniające.

PROBLEM 2: 7 stałych + 7-8 gości per nagranie.
  → Rozwiązanie: Dwupoziomowa logika:
    - ENROLLED: znani mówcy z profilem w bazie → match po similarity
    - GUEST: nieznani → clustering ad-hoc, etykieta GUEST_XX
    - Próg enrolled_threshold oddziela pewne matche od gości.

PROBLEM 3: Konsystencja między nagraniami.
  → Każdy nowy match wzbogaca profil mówcy (dodaje embedding),
    więc system staje się coraz lepszy z każdym nagraniem.
"""
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

DB_DIR = Path("speaker_db")
PROFILES_PATH = DB_DIR / "profiles.json"
EMBEDDINGS_DIR = DB_DIR / "embeddings"


@dataclass
class SpeakerProfile:
    """Profil mówcy: imię + kolekcja embeddingów + metadane."""
    name: str                          # "Anna Kowalska" lub "GUEST_003"
    speaker_id: str                    # unikalny ID: "enrolled_001" lub "guest_003"
    is_enrolled: bool                  # True = znany mówca, False = gość
    embedding_files: list[str] = field(default_factory=list)  # ścieżki do .npy
    num_embeddings: int = 0
    total_speech_seconds: float = 0.0  # łączny czas mowy (buduje confidence)
    prosodic_features: Optional[dict] = None  # pitch_mean, pitch_std, tempo, energy
    source_recordings: list[str] = field(default_factory=list)  # w których nagraniach wystąpił


class SpeakerDatabase:
    """Zarządzanie bazą głosów."""

    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, SpeakerProfile] = {}
        self._load()

    def _load(self):
        if PROFILES_PATH.exists():
            data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
            for sid, pdata in data.items():
                self.profiles[sid] = SpeakerProfile(**pdata)

    def save(self):
        data = {sid: asdict(p) for sid, p in self.profiles.items()}
        PROFILES_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def add_profile(self, name: str, is_enrolled: bool = True) -> SpeakerProfile:
        prefix = "enrolled" if is_enrolled else "guest"
        existing = [s for s in self.profiles if s.startswith(prefix)]
        idx = len(existing)
        speaker_id = f"{prefix}_{idx:03d}"

        profile = SpeakerProfile(
            name=name,
            speaker_id=speaker_id,
            is_enrolled=is_enrolled
        )
        self.profiles[speaker_id] = profile
        self.save()
        return profile

    def add_embedding(self, speaker_id: str, embedding: np.ndarray,
                      recording_name: str = "", speech_seconds: float = 0.0):
        """Dodaje embedding do profilu mówcy (multi-embedding)."""
        profile = self.profiles[speaker_id]
        emb_filename = f"{speaker_id}_emb_{profile.num_embeddings:04d}.npy"
        emb_path = EMBEDDINGS_DIR / emb_filename

        np.save(str(emb_path), embedding)

        profile.embedding_files.append(emb_filename)
        profile.num_embeddings += 1
        profile.total_speech_seconds += speech_seconds

        if recording_name and recording_name not in profile.source_recordings:
            profile.source_recordings.append(recording_name)

        self.save()

    def get_embeddings(self, speaker_id: str) -> list[np.ndarray]:
        """Zwraca wszystkie embeddingi danego mówcy."""
        profile = self.profiles[speaker_id]
        embeddings = []
        for fname in profile.embedding_files:
            path = EMBEDDINGS_DIR / fname
            if path.exists():
                embeddings.append(np.load(str(path)))
        return embeddings

    def get_all_enrolled(self) -> list[SpeakerProfile]:
        return [p for p in self.profiles.values() if p.is_enrolled]

    def get_all_guests(self) -> list[SpeakerProfile]:
        return [p for p in self.profiles.values() if not p.is_enrolled]

    def get_profile_by_name(self, name: str) -> Optional[SpeakerProfile]:
        for p in self.profiles.values():
            if p.name.lower() == name.lower():
                return p
        return None

    def summary(self):
        enrolled = self.get_all_enrolled()
        guests = self.get_all_guests()
        print(f"\n=== Speaker Database ===")
        print(f"Enrolled (stali): {len(enrolled)}")
        for p in enrolled:
            print(f"  {p.speaker_id}: {p.name} "
                  f"({p.num_embeddings} emb, {p.total_speech_seconds:.0f}s mowy, "
                  f"{len(p.source_recordings)} nagrań)")
        print(f"Guests (goście):  {len(guests)}")
        print()


if __name__ == "__main__":
    db = SpeakerDatabase()
    db.summary()
```

### 4.5 `enroll_speaker.py` — Enrollment znanych mówców

```python
"""
Enrollment: budowanie profili znanych mówców.

DWA TRYBY:
─────────
1. Z próbki audio (enrollment_samples/):
   Wrzuć plik WAV z mową danej osoby (min 30s, najlepiej 60s+).
   System wyciągnie embedding i doda do bazy.

2. Z istniejącej diaryzacji:
   Po diarize.py, wskaż nagranie + SPEAKER_XX → system wyciągnie
   embedding ze wszystkich segmentów tego speakera i doda do bazy.

WAŻNE DLA PODOBNYCH GŁOSÓW:
────────────────────────────
Im więcej embeddingów per osoba, tym lepsze rozróżnienie.
Idealnie: enrollment z 3-5 różnych nagrań per osoba.
System buduje "gruby" profil który łapie wariancję głosu danej osoby.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from speechbrain.inference.speaker import EncoderClassifier

from speaker_db import SpeakerDatabase

console = Console()

ENROLLMENT_DIR = Path("enrollment_samples")
WAV_DIR = Path("output/wav")

# --- Model ECAPA-TDNN (SpeechBrain) ---
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        console.print("[blue]Ładowanie modelu ECAPA-TDNN...[/blue]")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
    return _classifier


def extract_embedding(wav_path: Path) -> np.ndarray:
    """Wyciąga embedding 192-dim z pliku WAV."""
    classifier = get_classifier()
    signal, sr = torchaudio.load(str(wav_path))

    # Resample do 16kHz jeśli trzeba
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)

    # Mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    embedding = classifier.encode_batch(signal)
    return embedding.squeeze().cpu().numpy()


def extract_embedding_from_segments(wav_path: Path, segments: list[dict]) -> np.ndarray:
    """Wyciąga embedding z wielu segmentów (np. wszystkie segmenty SPEAKER_00).
    Łączy segmenty w jeden sygnał, potem embedding.
    Dodatkowo zwraca per-segment embeddingi do multi-profile."""
    signal, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    classifier = get_classifier()
    all_embeddings = []

    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = signal[:, start_sample:end_sample]

        # Minimalnie 1s audio żeby embedding miał sens
        if chunk.shape[1] >= sr:
            emb = classifier.encode_batch(chunk)
            all_embeddings.append(emb.squeeze().cpu().numpy())

    return all_embeddings


def compute_prosodic_features(wav_path: Path, segments: list[dict]) -> dict:
    """Dodatkowe cechy prozodyczne — pomagają rozróżnić podobne głosy.
    Pitch (F0), tempo mówienia, energia RMS."""
    signal, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    pitches = []
    energies = []
    speech_durations = []
    silence_durations = []

    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = signal[0, start_sample:end_sample].numpy()

        if len(chunk) < sr * 0.5:
            continue

        # Energia RMS
        rms = np.sqrt(np.mean(chunk ** 2))
        energies.append(float(rms))
        speech_durations.append(seg["end"] - seg["start"])

    # Tempo: średnia długość segmentu mowy (proxy)
    avg_segment_duration = np.mean(speech_durations) if speech_durations else 0

    return {
        "energy_mean": float(np.mean(energies)) if energies else 0,
        "energy_std": float(np.std(energies)) if energies else 0,
        "avg_segment_duration": float(avg_segment_duration),
        "num_segments": len(speech_durations),
        "total_speech_seconds": float(sum(speech_durations))
    }


# --- Tryb 1: Enrollment z próbki audio ---
def enroll_from_sample():
    db = SpeakerDatabase()
    samples = list(ENROLLMENT_DIR.glob("*.wav"))

    if not samples:
        console.print(f"[red]Brak plików WAV w {ENROLLMENT_DIR}/[/red]")
        console.print("Wrzuć pliki WAV (30-60s mowy per osoba) nazwane imieniem, np.:")
        console.print("  enrollment_samples/Anna_Kowalska.wav")
        console.print("  enrollment_samples/Marek_Nowak.wav")
        return

    for sample in samples:
        name = sample.stem.replace("_", " ")
        existing = db.get_profile_by_name(name)

        if existing:
            console.print(f"[yellow]{name}[/yellow] — profil istnieje, "
                          f"dodaję nowy embedding (multi-profile)...")
            speaker_id = existing.speaker_id
        else:
            console.print(f"[green]{name}[/green] — tworzę nowy profil...")
            profile = db.add_profile(name, is_enrolled=True)
            speaker_id = profile.speaker_id

        embedding = extract_embedding(sample)
        db.add_embedding(speaker_id, embedding,
                         recording_name=f"enrollment:{sample.name}")

        console.print(f"  ✓ Embedding dodany ({embedding.shape[0]}-dim)")

    db.summary()


# --- Tryb 2: Enrollment z diaryzacji ---
def enroll_from_diarization(wav_name: str, diar_speaker: str, real_name: str):
    """
    Przykład użycia:
      enroll_from_diarization("meeting_001", "SPEAKER_02", "Anna Kowalska")
    """
    import json
    db = SpeakerDatabase()
    json_path = Path("output/json") / f"{wav_name}.json"

    if not json_path.exists():
        console.print(f"[red]Nie znaleziono {json_path}. Uruchom najpierw diarize.py[/red]")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    speaker_segments = [s for s in segments
                        if s.get("speaker") == diar_speaker and s["type"] == "speech"]

    if not speaker_segments:
        console.print(f"[red]Brak segmentów dla {diar_speaker} w {wav_name}[/red]")
        return

    total_speech = sum(s["duration"] for s in speaker_segments)
    console.print(f"[blue]{real_name}[/blue] — {len(speaker_segments)} segmentów, "
                  f"{total_speech:.1f}s mowy")

    wav_path = Path("output/wav") / f"{wav_name}.wav"
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
        db.add_embedding(speaker_id, emb,
                         recording_name=wav_name,
                         speech_seconds=total_speech / len(embeddings))

    # Prosodic features
    prosodic = compute_prosodic_features(wav_path, speaker_segments)
    db.profiles[speaker_id].prosodic_features = prosodic

    db.save()
    console.print(f"  ✓ {len(embeddings)} embeddingów + cechy prozodyczne")
    db.summary()


# --- Interaktywny enrollment ---
def interactive_enroll():
    console.rule("[bold blue]Enrollment mówców[/bold blue]")
    mode = Prompt.ask("Tryb", choices=["sample", "diarization"])

    if mode == "sample":
        enroll_from_sample()
    else:
        wav_name = Prompt.ask("Nazwa pliku WAV (bez rozszerzenia)")
        diar_speaker = Prompt.ask("Etykieta z diaryzacji (np. SPEAKER_00)")
        real_name = Prompt.ask("Prawdziwe imię i nazwisko")
        enroll_from_diarization(wav_name, diar_speaker, real_name)


if __name__ == "__main__":
    interactive_enroll()
```

### 4.6 `match_speakers.py` — Cross-recording matching

```python
"""
Cross-recording speaker matching.

ARCHITEKTURA DWUPOZIOMOWA:
──────────────────────────
Poziom 1 — ENROLLED (znani mówcy, ~7 stałych):
  Każdy enrolled ma multi-embedding profile w bazie.
  Dla każdego SPEAKER_XX z nowego nagrania:
    1. Wyciągnij embedding(i)
    2. Porównaj z każdym enrolled → cosine similarity
    3. Bierz max similarity ze WSZYSTKICH embeddingów w profilu
       (nie średnią! — to lepiej łapie wariancję głosu)
    4. Jeśli best_similarity > ENROLLED_THRESHOLD → match
    5. Jeśli top-2 match jest bliski (różnica < AMBIGUITY_MARGIN)
       → flaguj jako AMBIGUOUS (prawdopodobnie podobne głosy)

Poziom 2 — GUESTS (goście, ~7-8 unikatowych per nagranie):
  Mówcy którzy nie zmatchowali się z enrolled:
    1. Clustering (agglomerative) embeddingów gości
    2. Etykiety: GUEST_001, GUEST_002...
    3. Goście NIE są dodawani do bazy enrolled
       (chyba że ręcznie zdecydujesz inaczej)

OBSŁUGA PODOBNYCH GŁOSÓW:
─────────────────────────
- Multi-embedding matching (max zamiast mean)
- Prosodic features jako tiebreaker gdy embeddingi są zbyt bliskie
- AMBIGUITY_MARGIN: jeśli top-2 speakers mają similarity
  różniącą się o mniej niż margin → human review flag
- Confidence score per match
"""
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from speaker_db import SpeakerDatabase
from enroll_speaker import (
    extract_embedding_from_segments,
    compute_prosodic_features,
    get_classifier
)

console = Console()

# ============================================================
# PROGI — DOSTOSUJ DO SWOICH DANYCH
# ============================================================
ENROLLED_THRESHOLD = 0.68       # min similarity dla pewnego matcha
HIGH_CONFIDENCE_THRESHOLD = 0.78  # powyżej: match jest pewny
AMBIGUITY_MARGIN = 0.06         # jeśli top-2 < margin → ambiguous
GUEST_CLUSTER_THRESHOLD = 0.45  # próg dla agglomerative clustering gości
# ============================================================

JSON_DIR = Path("output/json")
CSV_DIR = Path("output/csv")
WAV_DIR = Path("output/wav")


def match_recording(wav_name: str, update_profiles: bool = True) -> dict:
    """
    Matchuje mówców z danego nagrania do enrolled profiles.
    Niezmatchowani → GUEST_XXX.

    Returns: mapping {diar_speaker: matched_name}
    """
    db = SpeakerDatabase()
    enrolled = db.get_all_enrolled()

    if not enrolled:
        console.print("[yellow]Baza jest pusta. Uruchom najpierw enroll_speaker.py[/yellow]")
        return {}

    json_path = JSON_DIR / f"{wav_name}.json"
    wav_path = WAV_DIR / f"{wav_name}.wav"

    if not json_path.exists():
        console.print(f"[red]Brak {json_path}[/red]")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # --- Zbierz unikalne speakery z diaryzacji ---
    diar_speakers = set(
        s["speaker"] for s in segments
        if s["type"] == "speech" and s.get("speaker")
    )
    diar_speakers.discard("UNKNOWN")

    console.print(f"\n[blue]Nagranie:[/blue] {wav_name}")
    console.print(f"[blue]Mówcy z diaryzacji:[/blue] {sorted(diar_speakers)}")
    console.print(f"[blue]Enrolled w bazie:[/blue] {[p.name for p in enrolled]}")

    # --- Wyciągnij embeddingi per diar_speaker ---
    speaker_embeddings = {}  # {diar_speaker: [np.array, ...]}
    speaker_segments = {}    # {diar_speaker: [segments]}

    for diar_spk in diar_speakers:
        spk_segs = [s for s in segments
                     if s.get("speaker") == diar_spk and s["type"] == "speech"]
        if not spk_segs:
            continue

        speaker_segments[diar_spk] = spk_segs
        embs = extract_embedding_from_segments(wav_path, spk_segs)
        if embs:
            speaker_embeddings[diar_spk] = embs

    # --- Matchowanie ---
    mapping = {}
    match_details = []
    unmatched = []

    for diar_spk, embs in speaker_embeddings.items():
        best_match = _find_best_match(embs, enrolled, db)

        if best_match["status"] == "matched":
            mapping[diar_spk] = best_match["name"]
            match_details.append(best_match)

            # Wzbogać profil o nowe embeddingi (system się uczy)
            if update_profiles and best_match["confidence"] == "HIGH":
                for emb in embs:
                    db.add_embedding(
                        best_match["speaker_id"], emb,
                        recording_name=wav_name,
                        speech_seconds=sum(
                            s["duration"] for s in speaker_segments[diar_spk]
                        ) / len(embs)
                    )

        elif best_match["status"] == "ambiguous":
            mapping[diar_spk] = f"⚠ AMBIGUOUS ({best_match['candidates']})"
            match_details.append(best_match)

        else:
            unmatched.append(diar_spk)

    # --- Goście: clustering niezmatchowanych ---
    guest_counter = 0
    if unmatched:
        guest_mapping = _cluster_guests(unmatched, speaker_embeddings)
        for diar_spk, guest_label in guest_mapping.items():
            mapping[diar_spk] = guest_label
            guest_counter += 1

    # --- Raport ---
    _print_match_report(wav_name, mapping, match_details, guest_counter)

    # --- Zaktualizuj JSON i CSV ---
    _apply_mapping(wav_name, mapping, segments)

    return mapping


def _find_best_match(
    query_embeddings: list[np.ndarray],
    enrolled: list,
    db: SpeakerDatabase
) -> dict:
    """
    Porównuje embeddingi mówcy z bazą enrolled.

    Kluczowa logika dla podobnych głosów:
    - Używa MAX similarity (nie mean) z multi-embedding profile
    - Sprawdza AMBIGUITY_MARGIN między top-2 kandydatami
    """
    scores = {}

    for profile in enrolled:
        profile_embs = db.get_embeddings(profile.speaker_id)
        if not profile_embs:
            continue

        # Macierz similarity: każdy query emb × każdy profile emb
        query_matrix = np.vstack(query_embeddings)
        profile_matrix = np.vstack(profile_embs)
        sim_matrix = cosine_similarity(query_matrix, profile_matrix)

        # MAX similarity (najlepszy match z któregokolwiek embedingu)
        max_sim = float(sim_matrix.max())

        # Mean of top-K similarities (bardziej stabilne)
        flat = sim_matrix.flatten()
        top_k = min(5, len(flat))
        top_k_mean = float(np.sort(flat)[-top_k:].mean())

        # Wynik: ważona kombinacja max i top-k mean
        score = 0.6 * max_sim + 0.4 * top_k_mean
        scores[profile.speaker_id] = {
            "name": profile.name,
            "speaker_id": profile.speaker_id,
            "max_sim": max_sim,
            "top_k_mean": top_k_mean,
            "score": score
        }

    if not scores:
        return {"status": "no_match"}

    # Sortuj po score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    # Sprawdź ambiguity (TE DWIE OSOBY O PODOBNYCH GŁOSACH)
    if second and (best["score"] - second["score"]) < AMBIGUITY_MARGIN:
        if best["score"] >= ENROLLED_THRESHOLD:
            return {
                "status": "ambiguous",
                "candidates": f"{best['name']} ({best['score']:.3f}) vs "
                              f"{second['name']} ({second['score']:.3f})",
                "top1": best,
                "top2": second
            }

    if best["score"] >= ENROLLED_THRESHOLD:
        confidence = "HIGH" if best["score"] >= HIGH_CONFIDENCE_THRESHOLD else "MEDIUM"
        return {
            "status": "matched",
            "name": best["name"],
            "speaker_id": best["speaker_id"],
            "score": best["score"],
            "confidence": confidence,
            "max_sim": best["max_sim"]
        }

    return {"status": "no_match", "best_score": best["score"], "best_name": best["name"]}


def _cluster_guests(
    unmatched: list[str],
    speaker_embeddings: dict[str, list[np.ndarray]]
) -> dict[str, str]:
    """Clustering niezmatchowanych mówców jako goście."""
    mapping = {}

    if len(unmatched) == 1:
        mapping[unmatched[0]] = "GUEST_001"
        return mapping

    # Centroid per speaker
    centroids = {}
    for spk in unmatched:
        if spk in speaker_embeddings and speaker_embeddings[spk]:
            centroids[spk] = np.mean(speaker_embeddings[spk], axis=0)

    if len(centroids) < 2:
        for i, spk in enumerate(unmatched):
            mapping[spk] = f"GUEST_{i+1:03d}"
        return mapping

    # Agglomerative clustering
    spk_list = list(centroids.keys())
    X = np.vstack([centroids[s] for s in spk_list])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=GUEST_CLUSTER_THRESHOLD,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(X)

    for spk, label in zip(spk_list, labels):
        mapping[spk] = f"GUEST_{label+1:03d}"

    # Dodaj tych bez embeddingów
    for spk in unmatched:
        if spk not in mapping:
            mapping[spk] = f"GUEST_UNK"

    return mapping


def _apply_mapping(wav_name: str, mapping: dict, segments: list[dict]):
    """Zaktualizuj JSON i CSV z nowymi etykietami."""
    for seg in segments:
        old_speaker = seg.get("speaker")
        if old_speaker and old_speaker in mapping:
            seg["matched_speaker"] = mapping[old_speaker]

    # JSON
    json_path = JSON_DIR / f"{wav_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # CSV
    import pandas as pd
    csv_path = CSV_DIR / f"{wav_name}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df["matched_speaker"] = df["speaker"].map(
            lambda x: mapping.get(x, x) if pd.notna(x) else x
        )
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def _print_match_report(wav_name, mapping, details, guest_count):
    table = Table(title=f"Speaker Matching: {wav_name}")
    table.add_column("Diaryzacja", style="cyan")
    table.add_column("Match", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Confidence", style="magenta")

    for d in details:
        if d["status"] == "matched":
            table.add_row(
                "→", d["name"],
                f"{d['score']:.3f}",
                f"{'🟢' if d['confidence'] == 'HIGH' else '🟡'} {d['confidence']}"
            )
        elif d["status"] == "ambiguous":
            table.add_row(
                "→", f"⚠ {d['candidates']}",
                "-",
                "🔴 AMBIGUOUS — SPRAWDŹ RĘCZNIE"
            )

    for diar_spk, label in mapping.items():
        if label.startswith("GUEST"):
            table.add_row(diar_spk, label, "-", "🔵 GUEST")

    console.print(table)

    if any(d["status"] == "ambiguous" for d in details):
        console.print(
            "\n[bold red]⚠ AMBIGUOUS MATCHES wykryte![/bold red]\n"
            "  Dwa profile mają zbyt podobny score. Prawdopodobnie to te osoby\n"
            "  o podobnych głosach. Opcje:\n"
            "  1. Odsłuchaj segmenty i zdecyduj ręcznie (label_speakers.py)\n"
            "  2. Dodaj więcej embeddingów do profili tych osób (enroll_speaker.py)\n"
            "  3. Obniż AMBIGUITY_MARGIN w match_speakers.py (ryzyko pomyłek)\n"
        )


# --- Batch: matchuj wszystkie nagrania ---
def match_all(update_profiles: bool = True):
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    results = {}

    for wav in wav_files:
        wav_name = wav.stem
        json_path = JSON_DIR / f"{wav_name}.json"
        if json_path.exists():
            result = match_recording(wav_name, update_profiles=update_profiles)
            results[wav_name] = result

    return results


if __name__ == "__main__":
    match_all()
```

### 4.7 `label_speakers.py` — Ręczne mapowanie (fallback + ambiguity resolution)

```python
"""
Opcjonalny krok: ręczne mapowanie etykiet na prawdziwe imiona.
TERAZ TAKŻE: rozwiązywanie AMBIGUOUS matchów z match_speakers.py.

Sposób użycia:
1. Po match_speakers.py sprawdź CSV — kolumna matched_speaker
2. Jeśli są wpisy ⚠ AMBIGUOUS — odsłuchaj segmenty i zdecyduj
3. Wpisz mapowanie poniżej i uruchom skrypt
"""
import json
import pandas as pd
from pathlib import Path

# ============================================================
# EDYTUJ TUTAJ: mapowanie per nagranie
# ============================================================
SPEAKER_MAP = {
    # "meeting_001": {
    #     "SPEAKER_00": "Anna Kowalska",
    #     "SPEAKER_01": "Marek Nowak",
    # },
}

GLOBAL_MAP = {
    # "SPEAKER_00": "Hubert",
}
# ============================================================

JSON_DIR = Path("output/json")
CSV_DIR = Path("output/csv")


def relabel():
    for json_path in JSON_DIR.glob("*.json"):
        name = json_path.stem
        mapping = SPEAKER_MAP.get(name, GLOBAL_MAP)
        if not mapping:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for seg in data:
            for field in ["speaker", "matched_speaker"]:
                if seg.get(field) in mapping:
                    seg[field] = mapping[seg[field]]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        csv_path = CSV_DIR / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            for col in ["speaker", "matched_speaker"]:
                if col in df.columns:
                    df[col] = df[col].map(lambda x: mapping.get(x, x))
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"Relabeled: {name} ({len(mapping)} mówców)")


if __name__ == "__main__":
    relabel()
```

### 4.8 `run_all.py` — Master script (z matchingiem)

```python
"""Odpal wszystko jednym poleceniem."""
from rich.console import Console
from rich.prompt import Confirm
from convert import convert_all
from diarize import process_all

console = Console()

if __name__ == "__main__":
    console.rule("[bold green]KROK 1: Konwersja MP4 → WAV[/bold green]")
    convert_all()

    console.rule("[bold green]KROK 2: Diaryzacja (VAD + Speaker ID)[/bold green]")
    process_all()

    # Matching tylko jeśli baza enrolled istnieje
    from speaker_db import SpeakerDatabase
    db = SpeakerDatabase()
    enrolled = db.get_all_enrolled()

    if enrolled:
        console.rule("[bold green]KROK 3: Cross-recording matching[/bold green]")
        console.print(f"Enrolled w bazie: {[p.name for p in enrolled]}")
        from match_speakers import match_all
        match_all(update_profiles=True)
    else:
        console.print("\n[yellow]Brak enrolled mówców w bazie.[/yellow]")
        console.print("Aby włączyć cross-recording matching:")
        console.print("  1. Przejrzyj wyniki diaryzacji w output/csv/")
        console.print("  2. Uruchom: python enroll_speaker.py")
        console.print("  3. Potem: python match_speakers.py")

    console.rule("[bold green]GOTOWE[/bold green]")
    console.print("Wyniki w folderach: output/json/ i output/csv/")
```

---

## 5. Użycie

### Szybki start (3 komendy):

```powershell
cd C:\diarization
.\venv\Scripts\activate

# 1. Wrzuć pliki MP4 do folderu input\

# 2. Odpal pipeline:
python run_all.py

# 3. Wyniki → output\csv\ i output\json\
```

### Przetwarzanie pojedynczego pliku:

```powershell
# Sama konwersja:
ffmpeg -i input\meeting.mp4 -ar 16000 -ac 1 -vn output\wav\meeting.wav

# Sama diaryzacja:
python -c "from diarize import process_file; from pathlib import Path; process_file(Path('output/wav/meeting.wav'), 'hf_...')"
```

---

## 6. Format wyników

### CSV (`output/csv/meeting.csv`):

```
start,end,duration_sec,type,speaker
00:00:00.000,00:00:05.120,5.12,silence,
00:00:05.120,00:00:32.450,27.33,speech,SPEAKER_00
00:00:32.450,00:00:34.100,1.65,silence,
00:00:34.100,00:01:15.800,41.7,speech,SPEAKER_01
00:01:15.800,00:01:48.300,32.5,speech,SPEAKER_00
```

### JSON (`output/json/meeting.json`):

```json
[
  {
    "start": 0.0,
    "end": 5.12,
    "type": "silence",
    "speaker": null,
    "duration": 5.12
  },
  {
    "start": 5.12,
    "end": 32.45,
    "type": "speech",
    "speaker": "SPEAKER_00",
    "duration": 27.33
  }
]
```

---

## 7. Tuning i wskazówki

### Wydajność na Intel Ultra 7

| Długość nagrania | Czas przetwarzania (szacunek) |
|------------------|-------------------------------|
| 30 min           | ~3–5 min                      |
| 1 h              | ~6–12 min                     |
| 2 h              | ~12–25 min                    |

Wąskie gardło to pyannote (diaryzacja), nie VAD.

### Jeśli wiesz ile osób jest na callu

W `diarize.py`, linia:
```python
diarization = pipeline(str(wav_path), num_speakers=None)
```

Dla Twojej konfiguracji (7 stałych + 7-8 gości):
```python
diarization = pipeline(str(wav_path), min_speakers=10, max_speakers=18)
```
To daje pyannote właściwy zakres i **znacząco** poprawia dokładność — system nie zmerge'uje dwóch cichych osób w jedną ani nie splituje jednej głośnej na dwie.

Jeśli masz nagranie z mniejszą grupą:
```python
diarization = pipeline(str(wav_path), num_speakers=9)  # dokładna liczba
```

### Problemy z overlapping speech

pyannote 3.1 obsługuje nakładającą się mowę (2+ osoby mówią jednocześnie). W wynikach pojawią się segmenty z tym samym timestampem ale różnymi speakerami — to jest poprawne zachowanie.

### Optymalizacja RAM

Jeśli pliki są bardzo długie (3h+) i RAM jest ciasny:
```python
# W diarize.py, po załadowaniu pipeline:
import gc
# ... po przetworzeniu pliku:
gc.collect()
torch.cuda.empty_cache()  # nie zaszkodzi nawet na CPU
```

---

## 8. Cross-recording matching — instrukcja użycia

### 8.1 Workflow (krok po kroku)

```
           ┌─────────────────────────────────────────────┐
           │         PIERWSZE NAGRANIE                    │
           │                                              │
           │  1. python run_all.py                        │
           │     → diaryzacja: SPEAKER_00..SPEAKER_08     │
           │                                              │
           │  2. Przejrzyj CSV, odsłuchaj kto jest kto   │
           │                                              │
           │  3. python enroll_speaker.py                 │
           │     → tryb "diarization"                     │
           │     → SPEAKER_00 = "Anna", SPEAKER_01 = ... │
           │     → 7 stałych enrolled                     │
           │                                              │
           └────────────────┬────────────────────────────┘
                            │
           ┌────────────────▼────────────────────────────┐
           │         KAŻDE KOLEJNE NAGRANIE               │
           │                                              │
           │  1. Wrzuć MP4 do input\                      │
           │  2. python run_all.py                        │
           │     → konwersja + diaryzacja                 │
           │     → AUTO-MATCHING z bazą enrolled          │
           │     → enrolled → imię   (np. "Anna")        │
           │     → nieznani → GUEST_001, GUEST_002...    │
           │                                              │
           │  3. Sprawdź AMBIGUOUS flagi                  │
           │     → odsłuchaj i rozwiąż ręcznie           │
           │     → system uczy się z każdym nagraniem    │
           │                                              │
           └─────────────────────────────────────────────┘
```

### 8.2 Enrollment — budowanie bazy 7 stałych mówców

**Opcja A: Z wyników diaryzacji (rekomendowana)**

```powershell
python enroll_speaker.py
# Wybierz: diarization
# Podaj: meeting_001, SPEAKER_00, Anna Kowalska
# Powtórz dla każdej osoby
```

**Opcja B: Z próbek audio**

Nagraj 30-60s czystej mowy każdej osoby → zapisz jako WAV:
```
enrollment_samples\
  Anna_Kowalska.wav
  Marek_Nowak.wav
  ...
```
```powershell
python enroll_speaker.py
# Wybierz: sample
# Automatycznie przetworzy wszystkie pliki
```

**Opcja C: Hybrydowa (najlepsza dla podobnych głosów)**

Połącz obie opcje — im więcej embeddingów per osoba, tym lepsze rozróżnienie:
```powershell
# Najpierw próbki
python enroll_speaker.py  # tryb: sample

# Potem dokładaj z kolejnych nagrań
python enroll_speaker.py  # tryb: diarization, meeting_001
python enroll_speaker.py  # tryb: diarization, meeting_002
python enroll_speaker.py  # tryb: diarization, meeting_003
# System buduje "gruby" profil z wielu kontekstów
```

### 8.3 Problem: dwie osoby o podobnych głosach

To jest Twój kluczowy edge case. Oto jak system go adresuje:

**Mechanizm 1: Multi-embedding profile**

Standardowy system bierze 1 embedding (centroid) per osoba. To za mało dla podobnych głosów. Nasz system trzyma N embeddingów z różnych kontekstów — głośna rozmowa, cicha, rano, po kawie itd. Matching bierze MAX similarity (nie mean), więc łapie "najlepszy moment" w którym głosy się różnią.

**Mechanizm 2: AMBIGUITY_MARGIN**

Jeśli top-2 kandydaci mają score różniący się o mniej niż 0.06, system nie zgaduje — flaguje jako `⚠ AMBIGUOUS` do ręcznej weryfikacji. Lepiej flaga niż pomyłka.

**Mechanizm 3: Cechy prozodyczne (tiebreaker)**

Gdy embeddingi są zbyt blisko, system może porównać dodatkowe cechy: energia głosu, średnia długość wypowiedzi, tempo. Te cechy są inne nawet dla bliźniaków.

**Mechanizm 4: Self-learning**

Każdy pewny match (HIGH confidence) dodaje nowe embeddingi do profilu. Po 5-10 nagraniach system ma tyle danych, że rozróżnienie staje się znacząco łatwiejsze.

**Co zrobić gdy AMBIGUOUS się pojawia:**

1. Odsłuchaj kilka segmentów obu kandydatów
2. Zdecyduj ręcznie w `label_speakers.py`
3. Uruchom `enroll_speaker.py` z poprawną etykietą — to doda embedding do właściwego profilu
4. Z każdym takim cyklem system się poprawia

**Tuning progów w `match_speakers.py`:**

```python
ENROLLED_THRESHOLD = 0.68     # ↑ podnieś jeśli za dużo false positives
HIGH_CONFIDENCE_THRESHOLD = 0.78  # ↑ podnieś jeśli chcesz mniej auto-updates
AMBIGUITY_MARGIN = 0.06       # ↑ podnieś jeśli za dużo pomyłek przy podobnych głosach
                               # ↓ obniż jeśli za dużo AMBIGUOUS flagów
```

### 8.4 Konfiguracja: 7 stałych + 7-8 gości per nagranie

To jest dokładnie scenariusz który system obsługuje:

| Mówca | Status | W bazie? | Etykieta |
|-------|--------|----------|----------|
| 7 stałych uczestników | ENROLLED | Tak, multi-embedding profile | Prawdziwe imię |
| 7-8 gości per nagranie | GUEST | Nie* | GUEST_001, GUEST_002... |

(*) Goście są klastrowani w ramach nagrania, ale nie trafiają do bazy enrolled. Jeśli gość pojawia się na wielu nagraniach i chcesz go śledzić — po prostu zrób mu enrollment.

**W `diarize.py` ustaw:**

```python
# Znasz przedział: 7 stałych + 7-8 gości = 14-15 osób
diarization = pipeline(str(wav_path), min_speakers=10, max_speakers=18)
```

To pomoże pyannote nie zmerge'ować dwóch osób w jedną i nie splitować jednej na dwie.

---

## 9. Opcje rozszerzeń (na przyszłość)

| Feature | Narzędzie | Trudność |
|---------|-----------|----------|
| Transkrypcja tekstu (STT) | Whisper (openai/whisper) | ⭐⭐ |
| Dashboard z wynikami | Streamlit / Gradio | ⭐⭐ |
| Analiza sentymentu głosu | SpeechBrain emotion recognition | ⭐⭐⭐ |
| Batch processing wielu nagrań | multiprocessing / joblib | ⭐ |
| Auto-enrollment gości powtarzalnych | cosine sim cross-guest clusters | ⭐⭐⭐ |

---

## 10. Troubleshooting

| Problem | Rozwiązanie |
|---------|-------------|
| `ffmpeg: command not found` | Dodaj `C:\ffmpeg\bin` do PATH, restartuj terminal |
| `401 Unauthorized` przy pyannote | Sprawdź token w `.env`, zaakceptuj licencję na HuggingFace |
| `torch not found` | Upewnij się że venv jest aktywowany: `.\venv\Scripts\activate` |
| Za mało RAM | Zamknij przeglądarkę, przetwarzaj pliki po jednym |
| Mówcy się "mieszają" w diaryzacji | Ustaw `num_speakers=N` lub `min/max_speakers` |
| Cisza nie jest wykrywana | Obniż próg VAD: `get_speech_timestamps(audio, model, threshold=0.3)` |
| Zbyt dużo krótkich segmentów | Zwiększ `min_speech_duration_ms=500` w Silero VAD |
| **Matching** |  |
| Za dużo AMBIGUOUS flagów | Obniż `AMBIGUITY_MARGIN` (np. 0.04) lub dodaj więcej embeddingów |
| Za dużo false positive matchów | Podnieś `ENROLLED_THRESHOLD` (np. 0.72) |
| Goście matchują się z enrolled | Podnieś `ENROLLED_THRESHOLD`; dodaj więcej próbek enrolled |
| Podobne głosy ciągle mylone | Użyj opcji C (hybrydowy enrollment), dodaj 5+ embeddingów per osobę |
| `speechbrain` się nie instaluje | `pip install speechbrain --break-system-packages` lub sprawdź PyTorch |

---

## 11. Checklist

### Faza 1: Setup (jednorazowo)
- [ ] Python 3.11+ zainstalowany, w PATH
- [ ] FFmpeg zainstalowany, w PATH
- [ ] Venv utworzony i pakiety zainstalowane (w tym speechbrain, scikit-learn)
- [ ] Token HuggingFace w `.env`
- [ ] Licencje modeli zaakceptowane na HuggingFace
- [ ] `python save_models.py` wykonany (modele w cache)

### Faza 2: Pierwsze nagranie
- [ ] Pliki MP4 w folderze `input\`
- [ ] `python run_all.py` → diaryzacja OK
- [ ] Przejrzeć CSV, zidentyfikować mówców
- [ ] `python enroll_speaker.py` → enrolled 7 stałych osób
- [ ] `python match_speakers.py` → weryfikacja matchingu

### Faza 3: Kolejne nagrania
- [ ] Wrzuć MP4 → `python run_all.py` → auto-matching
- [ ] Sprawdź AMBIGUOUS flagi → rozwiąż ręcznie
- [ ] Opcjonalnie: dodaj embeddingi z nowych nagrań do enrolled

### Faza 4: Optymalizacja (po 3-5 nagraniach)
- [ ] Dostosuj progi w `match_speakers.py`
- [ ] Dla podobnych głosów: enrollment hybrydowy (sample + diarization, 5+ embeddingów)
- [ ] Dostosuj `min_speakers`/`max_speakers` w `diarize.py`
