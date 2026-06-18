from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from funasr_timeline.asr.base import AsrToken
from funasr_timeline.normalization import normalize_text
from funasr_timeline.segmentation import SentenceSegment


@dataclass(frozen=True, slots=True)
class SentenceMatchResult:
    sentence_index: int
    match_score: float
    status: str
    matched_token_indexes: list[int]
    asr_token_range: tuple[int | None, int | None]
    matched_asr_text: str
    candidate_count: int
    selected_candidate_rank: int | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Candidate:
    start_pos: int
    end_pos: int
    text: str
    token_indexes: list[int]
    token_index_by_char: list[int]
    score: float


def match_sentences_to_tokens(
    segments: list[SentenceSegment],
    tokens: list[AsrToken],
    low_confidence_threshold: float = 0.8,
) -> list[SentenceMatchResult]:
    """Match manuscript sentences to ASR tokens in order using small fuzzy windows."""

    if not tokens:
        return [_empty_match(segment, "no_match") for segment in segments]

    token_pos_by_index = {token.index: position for position, token in enumerate(tokens)}
    cursor_pos = 0
    results: list[SentenceMatchResult] = []

    for segment in segments:
        if not segment.normalized_text:
            results.append(_empty_match(segment, "empty_after_normalization"))
            continue

        candidates = _build_candidates(
            segment=segment,
            tokens=tokens,
            cursor_pos=cursor_pos,
            max_token_pos=len(tokens),
        )
        if not candidates:
            results.append(_empty_match(segment, "no_match"))
            continue

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                -abs(len(candidate.text) - len(segment.normalized_text)),
                -candidate.start_pos,
            ),
            reverse=True,
        )
        selected = ranked[0]
        query_text = _normalize_match_text(segment.normalized_text)
        matched_token_indexes, unmatched_chars = _matched_tokens_in_candidate(
            query_text,
            selected.text,
            selected.token_index_by_char,
        )
        exact_matched_token_indexes = _unique_in_order(matched_token_indexes)

        if selected.score <= 0:
            timeline_token_indexes: list[int] = []
        else:
            timeline_token_indexes = selected.token_indexes

        if timeline_token_indexes:
            asr_token_range: tuple[int | None, int | None] = (
                timeline_token_indexes[0],
                timeline_token_indexes[-1],
            )
        else:
            asr_token_range = (None, None)

        status = "ok"
        if selected.score <= 0:
            status = "no_match"
        elif selected.score < low_confidence_threshold:
            status = "low_confidence"

        results.append(
            SentenceMatchResult(
                sentence_index=segment.index,
                match_score=round(selected.score, 6),
                status=status,
                matched_token_indexes=timeline_token_indexes,
                asr_token_range=asr_token_range,
                matched_asr_text=selected.text,
                candidate_count=len(candidates),
                selected_candidate_rank=1,
                diagnostics={
                    "text_similarity": round(selected.score, 6),
                    "matched_chars": len(query_text) - len(unmatched_chars),
                    "total_normalized_chars": len(query_text),
                    "exact_matched_token_indexes": exact_matched_token_indexes,
                    "unmatched_manuscript_chars": unmatched_chars,
                    "candidate_window": {
                        "start_token_index": selected.token_indexes[0],
                        "end_token_index": selected.token_indexes[-1],
                    },
                },
            )
        )

        if asr_token_range[1] is not None:
            cursor_pos = token_pos_by_index[asr_token_range[1]] + 1

    return results


def _build_candidates(
    segment: SentenceSegment,
    tokens: list[AsrToken],
    cursor_pos: int,
    max_token_pos: int,
) -> list[_Candidate]:
    query = _normalize_match_text(segment.normalized_text)
    query_length = len(query)
    min_chars = max(1, math.floor(query_length * 0.6))
    max_chars = max(min_chars, math.ceil(query_length * 1.5))
    search_limit = min(max_token_pos, cursor_pos + math.ceil(query_length * 2.5) + 10)
    start_limit = min(search_limit, cursor_pos + max(3, math.ceil(query_length * 0.4)))
    candidates: list[_Candidate] = []

    for start_pos in range(cursor_pos, start_limit):
        candidate_text = ""
        token_index_by_char: list[int] = []
        token_indexes: list[int] = []

        for end_pos in range(start_pos, search_limit):
            token = tokens[end_pos]
            normalized_token_text, normalized_token_chars = _normalize_token(token)
            if not normalized_token_text:
                continue

            candidate_text += normalized_token_text
            token_index_by_char.extend([token.index] * len(normalized_token_chars))
            token_indexes.append(token.index)
            candidate_length = len(candidate_text)

            if candidate_length < min_chars:
                continue
            if candidate_length > max_chars:
                break

            score = SequenceMatcher(
                a=query,
                b=_normalize_match_text(candidate_text),
                autojunk=False,
            ).ratio()
            candidates.append(
                _Candidate(
                    start_pos=start_pos,
                    end_pos=end_pos + 1,
                    text=candidate_text,
                    token_indexes=token_indexes.copy(),
                    token_index_by_char=token_index_by_char.copy(),
                    score=score,
                )
            )

    return candidates


def _normalize_token(token: AsrToken) -> tuple[str, list[str]]:
    normalized = normalize_text(token.text).text
    return normalized, list(normalized)


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\d+", _arabic_number_to_chinese_match_text, text)


def _arabic_number_to_chinese_match_text(match: re.Match[str]) -> str:
    value_text = match.group(0)
    if len(value_text) > 2:
        return "".join(_DIGIT_READINGS[int(char)] for char in value_text)

    value = int(value_text)
    if value < 10:
        return _DIGIT_READINGS[value]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + _DIGIT_READINGS[value % 10]
    tens, ones = divmod(value, 10)
    return _DIGIT_READINGS[tens] + "十" + (_DIGIT_READINGS[ones] if ones else "")


_DIGIT_READINGS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")


def _matched_tokens_in_candidate(
    query: str,
    candidate_text: str,
    token_index_by_char: list[int],
) -> tuple[list[int], list[dict[str, object]]]:
    matcher = SequenceMatcher(a=query, b=candidate_text, autojunk=False)
    matched_tokens: list[int] = []
    matched_query_indexes: set[int] = set()

    for tag, query_start, query_end, candidate_start, candidate_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        for query_index, candidate_index in zip(
            range(query_start, query_end),
            range(candidate_start, candidate_end),
            strict=True,
        ):
            matched_query_indexes.add(query_index)
            matched_tokens.append(token_index_by_char[candidate_index])

    unmatched_chars = [
        {
            "normalized_index": index,
            "char": char,
        }
        for index, char in enumerate(query)
        if index not in matched_query_indexes
    ]
    return matched_tokens, unmatched_chars


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _empty_match(segment: SentenceSegment, status: str) -> SentenceMatchResult:
    total_chars = len(segment.normalized_text)
    return SentenceMatchResult(
        sentence_index=segment.index,
        match_score=0.0,
        status=status,
        matched_token_indexes=[],
        asr_token_range=(None, None),
        matched_asr_text="",
        candidate_count=0,
        selected_candidate_rank=None,
        diagnostics={
            "text_similarity": 0.0,
            "matched_chars": 0,
            "total_normalized_chars": total_chars,
            "unmatched_manuscript_chars": [
                {
                    "normalized_index": index,
                    "char": char,
                }
                for index, char in enumerate(segment.normalized_text)
            ],
        },
    )
