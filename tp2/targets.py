"""Utilitários para os alvos: seed a partir do nome, carregar e listar imagens."""
import re
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def seed_from_filename(path, fallback: int = 2026) -> int:
    """Lê o inteiro inicial do nome do ficheiro como seed de render.

    7836.png -> 7836 ; 1159_25.png -> 1159
    """
    match = re.match(r"^(\d+)", Path(path).stem)
    return int(match.group(1)) if match else fallback


def safe_stem(path) -> str:
    """Nome de ficheiro seguro para usar como pasta (só alfanumérico, - e _)."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(path).stem)


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_target_images(directory) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
