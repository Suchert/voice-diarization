"""
Baza głosów: przechowuje profile mówców jako kolekcje embeddingów.

Architektura:
─────────────
PROBLEM 1: Dwie osoby o podobnych głosach.
  → Multi-embedding profile. Zamiast jednego centroidu per osoba,
    trzymamy N embeddingów (z różnych nagrań, kontekstów, energii głosu).
    Matching: max/mean similarity do WSZYSTKICH embeddingów w profilu.

PROBLEM 2: 7 stałych + 7-8 gości per nagranie.
  → Dwupoziomowa logika:
    - ENROLLED: znani mówcy z profilem w bazie → match po similarity
    - GUEST: nieznani → clustering ad-hoc, etykieta GUEST_XX

PROBLEM 3: Konsystencja między nagraniami.
  → Każdy nowy match wzbogaca profil mówcy (dodaje embedding).
    System staje się coraz lepszy z każdym nagraniem.
"""
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import DB_DIR, EMBEDDINGS_DIR, PROFILES_PATH


@dataclass
class SpeakerProfile:
    """Profil mówcy: imię + kolekcja embeddingów + metadane."""
    name: str
    speaker_id: str
    is_enrolled: bool
    embedding_files: list[str] = field(default_factory=list)
    num_embeddings: int = 0
    total_speech_seconds: float = 0.0
    prosodic_features: Optional[dict] = None
    source_recordings: list[str] = field(default_factory=list)


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

    def add_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        recording_name: str = "",
        speech_seconds: float = 0.0
    ):
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
        print(f"\n{'='*50}")
        print(f" Speaker Database")
        print(f"{'='*50}")
        print(f" Enrolled (stali): {len(enrolled)}")
        for p in enrolled:
            print(f"   {p.speaker_id}: {p.name} "
                  f"({p.num_embeddings} emb, {p.total_speech_seconds:.0f}s mowy, "
                  f"{len(p.source_recordings)} nagrań)")
        print(f" Guests (goście):  {len(guests)}")
        print()


if __name__ == "__main__":
    db = SpeakerDatabase()
    db.summary()
