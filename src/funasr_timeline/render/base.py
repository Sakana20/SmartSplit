from __future__ import annotations

from typing import Protocol

from funasr_timeline.merge import SentenceTimelineItem


class TimelineRenderer(Protocol):
    name: str
    file_extension: str

    def render(self, items: list[SentenceTimelineItem]) -> str:
        """Render sentence timeline items to a text artifact."""
