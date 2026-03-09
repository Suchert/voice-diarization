"""
Jednorazowe pobranie modeli do cache'a.
Po uruchomieniu tego skryptu pipeline działa OFFLINE.

Uruchom raz z internetem:
  python save_models.py
"""
import win_compat  # noqa: F401 — monkey-patches os.symlink on Windows

import torch
from rich.console import Console
from config import load_hf_token, PYANNOTE_MODEL, ECAPA_MODEL, ECAPA_SAVEDIR

console = Console()


def main():
    hf_token = load_hf_token()
    console.rule("[bold blue]Pobieranie modeli (jednorazowo)[/bold blue]")

    # 1. pyannote diarization (~500 MB)
    console.print("[blue]1/3[/blue] pyannote speaker-diarization-3.1...")
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_MODEL,
        use_auth_token=hf_token
    )
    pipeline.to(torch.device("cpu"))
    console.print("  [green]✓[/green] pyannote models cached.")

    # 2. SpeechBrain ECAPA-TDNN (~25 MB)
    console.print("[blue]2/3[/blue] SpeechBrain ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source=ECAPA_MODEL,
        savedir=ECAPA_SAVEDIR,
        run_opts={"device": "cpu"},
    )
    console.print("  [green]✓[/green] ECAPA-TDNN model cached.")

    # 3. Silero VAD (~2 MB)
    console.print("[blue]3/3[/blue] Silero VAD...")
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    console.print("  [green]✓[/green] Silero VAD cached.")

    console.rule("[bold green]Gotowe — pipeline działa offline[/bold green]")


if __name__ == "__main__":
    main()
