"""
Windows compatibility: zamienia os.symlink na shutil.copy2.

SpeechBrain wewnętrznie używa symlinków do cache'owania modeli.
Windows bez Developer Mode nie pozwala tworzyć symlinków.
Ten moduł patcha os.symlink żeby kopiował zamiast linkował.

IMPORT JAKO PIERWSZY w każdym skrypcie używającym SpeechBrain:
  import win_compat  # monkey-patches os.symlink on Windows
"""
import os
import sys
import shutil

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if sys.platform == "win32":
    _original_symlink = os.symlink

    def _copy_instead_of_symlink(src, dst, target_is_directory=False, **kwargs):
        src, dst = str(src), str(dst)
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        except Exception:
            _original_symlink(src, dst, target_is_directory)

    os.symlink = _copy_instead_of_symlink
