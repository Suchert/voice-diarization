"""
Centralna konfiguracja projektu.
Wszystkie parametry do tuningu w jednym miejscu.
"""
from pathlib import Path

# ============================================================
# ŚCIEŻKI
# ============================================================
PROJECT_DIR = Path(__file__).parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
WAV_DIR = OUTPUT_DIR / "wav"
JSON_DIR = OUTPUT_DIR / "json"
CSV_DIR = OUTPUT_DIR / "csv"
DB_DIR = PROJECT_DIR / "speaker_db"
EMBEDDINGS_DIR = DB_DIR / "embeddings"
PROFILES_PATH = DB_DIR / "profiles.json"
ENROLLMENT_DIR = PROJECT_DIR / "enrollment_samples"
PRETRAINED_DIR = PROJECT_DIR / "pretrained_models"
ENV_PATH = PROJECT_DIR / ".env"

# ============================================================
# AUDIO
# ============================================================
SAMPLE_RATE = 16000          # Hz — optymalny dla speech models
CHANNELS = 1                 # mono

# ============================================================
# VAD (Silero)
# ============================================================
VAD_THRESHOLD = 0.5          # próg detekcji mowy (0.0–1.0, default 0.5)
                             # ↓ obniż jeśli cisza nie jest wykrywana
                             # ↑ podnieś jeśli szum jest klasyfikowany jako mowa
MIN_SPEECH_DURATION_MS = 250  # min długość segmentu mowy (ms)
MIN_SILENCE_DURATION_MS = 100 # min przerwa żeby uznać za ciszę (ms)

# ============================================================
# DIARYZACJA (pyannote)
# ============================================================
# Dla konfiguracji 7 stałych + 7-8 gości:
MIN_SPEAKERS = 10            # None = auto-detect
MAX_SPEAKERS = 18            # None = auto-detect
# Jeśli znasz dokładną liczbę, ustaw NUM_SPEAKERS:
NUM_SPEAKERS = None          # int lub None (nadpisuje min/max jeśli ustawione)

# ============================================================
# SPEAKER MATCHING
# ============================================================
ENROLLED_THRESHOLD = 0.68    # min similarity dla pewnego matcha enrolled
                             # ↑ podnieś jeśli za dużo false positives
HIGH_CONFIDENCE_THRESHOLD = 0.78  # powyżej: match jest pewny, auto-update profilu
AMBIGUITY_MARGIN = 0.06     # jeśli top-2 speakers mają score różniący się
                             # o mniej niż margin → AMBIGUOUS flag
                             # ↑ podnieś jeśli za dużo pomyłek (podobne głosy)
                             # ↓ obniż jeśli za dużo AMBIGUOUS flagów
GUEST_CLUSTER_THRESHOLD = 0.45   # próg distance dla clustering gości
TOP_K_SIMILARITY = 5         # ile top similarities brać do mean score
SCORE_WEIGHT_MAX = 0.6       # waga max similarity w final score
SCORE_WEIGHT_TOPK = 0.4      # waga top-k mean similarity

# ============================================================
# EMBEDDING
# ============================================================
MIN_SEGMENT_DURATION_S = 1.0  # min długość segmentu do ekstrakcji embeddingu
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAVEDIR = str(PRETRAINED_DIR / "spkrec-ecapa-voxceleb")

# ============================================================
# PYANNOTE
# ============================================================
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# ============================================================
# HELPERS
# ============================================================
def load_hf_token() -> str:
    """Wczytaj HuggingFace token z .env"""
    if ENV_PATH.exists():
        # Próbuj różne kodowania (PowerShell `>` zapisuje UTF-16 LE z BOM)
        for encoding in ["utf-8-sig", "utf-16", "utf-8"]:
            try:
                content = ENV_PATH.read_text(encoding=encoding)
                for line in content.splitlines():
                    line = line.strip().strip("\x00")  # usuń null bytes
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if token and token != "hf_TwojTokenTutaj":
                            return token
                break  # plik się odczytał, po prostu nie ma tokena
            except (UnicodeDecodeError, UnicodeError):
                continue

    raise ValueError(
        f"Brak HF_TOKEN w pliku {ENV_PATH}\n"
        f"Utwórz plik .env w VS Code (UTF-8) z linią: HF_TOKEN=hf_TwojTokenTutaj\n"
        f"LUB w PowerShell: Set-Content -Path .env -Value 'HF_TOKEN=hf_...' -Encoding UTF8"
    )


def ensure_dirs():
    """Utwórz wszystkie katalogi projektowe."""
    for d in [INPUT_DIR, WAV_DIR, JSON_DIR, CSV_DIR,
              DB_DIR, EMBEDDINGS_DIR, ENROLLMENT_DIR, PRETRAINED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
