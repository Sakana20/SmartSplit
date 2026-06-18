from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Manuscript:
    path: str
    text: str


def read_txt_manuscript(path: Path) -> Manuscript:
    if path.suffix.lower() != ".txt":
        raise ValueError(f"第一阶段仅支持 .txt 稿件：{path}")
    text = path.read_text(encoding="utf-8")
    return Manuscript(path=str(path), text=text)
