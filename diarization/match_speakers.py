"""
Cross-recording speaker matching.

ARCHITEKTURA DWUPOZIOMOWA:
──────────────────────────
Poziom 1 — ENROLLED (znani mówcy, ~7 stałych):
  Multi-embedding matching z bazą. MAX similarity (nie mean).
  Jeśli top-2 match za bliski → AMBIGUOUS flag.

Poziom 2 — GUESTS (goście, ~7-8 per nagranie):
  Agglomerative clustering, etykiety GUEST_XXX.

SELF-LEARNING:
  Każdy pewny match (HIGH confidence) dodaje embeddingi do profilu.

Użycie:
  python match_speakers.py              # matchuj wszystkie nagrania
  python match_speakers.py meeting_001  # matchuj jedno nagranie
"""
import win_compat  # noqa: F401 — monkey-patches os.symlink on Windows

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from config import (
    WAV_DIR, JSON_DIR, CSV_DIR,
    ENROLLED_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    AMBIGUITY_MARGIN, GUEST_CLUSTER_THRESHOLD,
    TOP_K_SIMILARITY, SCORE_WEIGHT_MAX, SCORE_WEIGHT_TOPK,
    ensure_dirs
)
from speaker_db import SpeakerDatabase
from enroll_speaker import extract_embedding_from_segments

console = Console()


# ─────────────────────────────────────────────
# Core matching
# ─────────────────────────────────────────────
def match_recording(wav_name: str, update_profiles: bool = True) -> dict:
    """
    Matchuje mówców z nagrania do enrolled profiles.
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

    # Unikalne speakery z diaryzacji
    diar_speakers = set(
        s["speaker"] for s in segments
        if s["type"] == "speech" and s.get("speaker") and s["speaker"] != "UNKNOWN"
    )

    console.print(f"\n[blue]Nagranie:[/blue] {wav_name}")
    console.print(f"[blue]Mówcy z diaryzacji:[/blue] {sorted(diar_speakers)}")
    console.print(f"[blue]Enrolled w bazie:[/blue] {[p.name for p in enrolled]}")

    # Embeddingi per diar_speaker
    speaker_embeddings = {}
    speaker_segments = {}

    for diar_spk in diar_speakers:
        spk_segs = [
            s for s in segments
            if s.get("speaker") == diar_spk and s["type"] == "speech"
        ]
        if not spk_segs:
            continue

        speaker_segments[diar_spk] = spk_segs
        embs = extract_embedding_from_segments(wav_path, spk_segs)
        if embs:
            speaker_embeddings[diar_spk] = embs

    # Matchowanie
    mapping = {}
    match_details = []
    unmatched = []

    for diar_spk, embs in speaker_embeddings.items():
        best_match = _find_best_match(embs, enrolled, db)

        if best_match["status"] == "matched":
            mapping[diar_spk] = best_match["name"]
            match_details.append({**best_match, "diar_speaker": diar_spk})

            # Self-learning: wzbogać profil
            if update_profiles and best_match["confidence"] == "HIGH":
                spk_segs = speaker_segments.get(diar_spk, [])
                total_speech = sum(s["duration"] for s in spk_segs)
                for emb in embs:
                    db.add_embedding(
                        best_match["speaker_id"], emb,
                        recording_name=wav_name,
                        speech_seconds=total_speech / max(len(embs), 1)
                    )

        elif best_match["status"] == "ambiguous":
            mapping[diar_spk] = f"⚠ AMBIGUOUS ({best_match['candidates']})"
            match_details.append({**best_match, "diar_speaker": diar_spk})

        else:
            unmatched.append(diar_spk)

    # Goście: clustering
    guest_counter = 0
    if unmatched:
        guest_mapping = _cluster_guests(unmatched, speaker_embeddings)
        for diar_spk, guest_label in guest_mapping.items():
            mapping[diar_spk] = guest_label
            guest_counter += 1

    # Raport
    _print_match_report(wav_name, mapping, match_details, guest_counter)

    # Zaktualizuj JSON i CSV
    _apply_mapping(wav_name, mapping, segments)

    return mapping


def _find_best_match(
    query_embeddings: list[np.ndarray],
    enrolled: list,
    db: SpeakerDatabase
) -> dict:
    """
    Porównuje embeddingi z bazą enrolled.
    MAX similarity + AMBIGUITY_MARGIN check.
    """
    scores = {}

    for profile in enrolled:
        profile_embs = db.get_embeddings(profile.speaker_id)
        if not profile_embs:
            continue

        query_matrix = np.vstack(query_embeddings)
        profile_matrix = np.vstack(profile_embs)
        sim_matrix = cosine_similarity(query_matrix, profile_matrix)

        max_sim = float(sim_matrix.max())

        flat = sim_matrix.flatten()
        top_k = min(TOP_K_SIMILARITY, len(flat))
        top_k_mean = float(np.sort(flat)[-top_k:].mean())

        score = SCORE_WEIGHT_MAX * max_sim + SCORE_WEIGHT_TOPK * top_k_mean

        scores[profile.speaker_id] = {
            "name": profile.name,
            "speaker_id": profile.speaker_id,
            "max_sim": max_sim,
            "top_k_mean": top_k_mean,
            "score": score
        }

    if not scores:
        return {"status": "no_match"}

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    # Ambiguity check (podobne głosy)
    if second and (best["score"] - second["score"]) < AMBIGUITY_MARGIN:
        if best["score"] >= ENROLLED_THRESHOLD:
            return {
                "status": "ambiguous",
                "candidates": (
                    f"{best['name']} ({best['score']:.3f}) vs "
                    f"{second['name']} ({second['score']:.3f})"
                ),
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

    return {
        "status": "no_match",
        "best_score": best["score"],
        "best_name": best["name"]
    }


def _cluster_guests(
    unmatched: list[str],
    speaker_embeddings: dict[str, list[np.ndarray]]
) -> dict[str, str]:
    """Clustering niezmatchowanych mówców jako goście."""
    mapping = {}

    if len(unmatched) == 1:
        mapping[unmatched[0]] = "GUEST_001"
        return mapping

    centroids = {}
    for spk in unmatched:
        if spk in speaker_embeddings and speaker_embeddings[spk]:
            centroids[spk] = np.mean(speaker_embeddings[spk], axis=0)

    if len(centroids) < 2:
        for i, spk in enumerate(unmatched):
            mapping[spk] = f"GUEST_{i+1:03d}"
        return mapping

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

    for spk in unmatched:
        if spk not in mapping:
            mapping[spk] = "GUEST_UNK"

    return mapping


# ─────────────────────────────────────────────
# Apply & Report
# ─────────────────────────────────────────────
def _apply_mapping(wav_name: str, mapping: dict, segments: list[dict]):
    """Zaktualizuj JSON i CSV z matched_speaker."""
    for seg in segments:
        old_speaker = seg.get("speaker")
        if old_speaker and old_speaker in mapping:
            seg["matched_speaker"] = mapping[old_speaker]

    json_path = JSON_DIR / f"{wav_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

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
        diar_spk = d.get("diar_speaker", "?")
        if d["status"] == "matched":
            icon = "🟢" if d["confidence"] == "HIGH" else "🟡"
            table.add_row(
                diar_spk, d["name"],
                f"{d['score']:.3f}",
                f"{icon} {d['confidence']}"
            )
        elif d["status"] == "ambiguous":
            table.add_row(
                diar_spk, f"⚠ {d['candidates']}",
                "-", "🔴 AMBIGUOUS"
            )

    for diar_spk, label in mapping.items():
        if label.startswith("GUEST"):
            table.add_row(diar_spk, label, "-", "🔵 GUEST")

    console.print(table)

    if any(d["status"] == "ambiguous" for d in details):
        console.print(
            "\n[bold red]⚠ AMBIGUOUS MATCHES wykryte![/bold red]\n"
            "  Dwa profile mają zbyt podobny score — prawdopodobnie podobne głosy.\n"
            "  Opcje:\n"
            "  1. Odsłuchaj segmenty → label_speakers.py (ręczna decyzja)\n"
            "  2. Dodaj więcej embeddingów → enroll_speaker.py\n"
            "  3. Zmień AMBIGUITY_MARGIN w config.py\n"
        )


# ─────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────
def match_all(update_profiles: bool = True):
    ensure_dirs()
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
    if len(sys.argv) > 1:
        # Pojedyncze nagranie
        match_recording(sys.argv[1])
    else:
        match_all()
