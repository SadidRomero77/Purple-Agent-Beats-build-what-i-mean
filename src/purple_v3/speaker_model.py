"""Speaker modeling for cross-round adaptation.

Tracks per-speaker patterns to learn whether a speaker uses conservative
(same color, fewer blocks) or liberal (different color, more blocks)
interpretations for ambiguous instructions.

After observing 2+ ambiguous rounds for a speaker, uses learned pattern
instead of asking questions (-5 penalty avoided).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

# Minimum ambiguous rounds before trusting speaker pattern
_MIN_ROUNDS_FOR_RELIABLE = 2
# Threshold for "reliable" (conservative) speaker
_RELIABLE_THRESHOLD = 0.6


@dataclass
class SpeakerStats:
    """Tracks a speaker's behavior on ambiguous instructions."""
    total_rounds: int = 0
    correct_rounds: int = 0
    ambiguous_rounds: int = 0
    # Track if conservative (same color/fewer blocks) was correct
    conservative_correct: int = 0
    conservative_total: int = 0

    @property
    def reliability_score(self) -> float:
        """Ratio of times conservative interpretation was correct."""
        if self.conservative_total == 0:
            return 0.5  # Unknown
        return self.conservative_correct / self.conservative_total

    @property
    def is_reliable(self) -> bool:
        """Speaker consistently uses conservative interpretations."""
        return (
            self.ambiguous_rounds >= _MIN_ROUNDS_FOR_RELIABLE
            and self.reliability_score > _RELIABLE_THRESHOLD
        )

    @property
    def is_known(self) -> bool:
        """We have enough data to make a prediction."""
        return self.ambiguous_rounds >= _MIN_ROUNDS_FOR_RELIABLE


class SpeakerModel:
    """Track per-speaker patterns for ambiguity resolution."""

    def __init__(self):
        self._stats: Dict[str, SpeakerStats] = {}

    def _get_stats(self, speaker: str) -> SpeakerStats:
        if speaker not in self._stats:
            self._stats[speaker] = SpeakerStats()
        return self._stats[speaker]

    def record_feedback(self, speaker: str, was_correct: bool) -> None:
        """Record outcome of a round for this speaker."""
        if not speaker:
            return
        stats = self._get_stats(speaker)
        stats.total_rounds += 1
        if was_correct:
            stats.correct_rounds += 1
        logger.debug("Speaker %s: %d/%d correct", speaker, stats.correct_rounds, stats.total_rounds)

    def record_ambiguous_outcome(
        self, speaker: str, was_correct: bool, used_conservative: bool
    ) -> None:
        """Record outcome of an ambiguous round."""
        if not speaker:
            return
        stats = self._get_stats(speaker)
        stats.ambiguous_rounds += 1
        stats.conservative_total += 1
        if was_correct and used_conservative:
            stats.conservative_correct += 1
        logger.debug(
            "Speaker %s ambiguous: reliable=%.2f (%d/%d)",
            speaker, stats.reliability_score,
            stats.conservative_correct, stats.conservative_total,
        )

    def should_ask(self, speaker: str, ambiguity_type: str) -> bool:
        """Whether to ask this speaker about this type of ambiguity.

        Returns True if we should ask (don't know speaker well enough).
        Returns False if we know the speaker's pattern (save -5 penalty).
        """
        if not speaker:
            return True  # Unknown speaker → ask

        stats = self._get_stats(speaker)

        if not stats.is_known:
            return True  # Not enough data → ask to learn

        # Known speaker: don't ask, use learned pattern
        return False

    def get_interpretation(self, speaker: str, ambiguity_type: str) -> str:
        """Get the preferred interpretation for this speaker.

        Returns "conservative" or "liberal".
        Conservative = same color as existing, fewer blocks.
        Liberal = different color, more blocks.
        """
        if not speaker:
            return "conservative"  # Default: conservative

        stats = self._get_stats(speaker)

        if stats.is_reliable:
            return "conservative"

        # Unknown or unreliable → conservative is safer (higher base rate)
        return "conservative"

    def get_context_hint(self, speaker: str) -> str:
        """Get a context hint about this speaker for the planner prompt."""
        if not speaker:
            return ""

        stats = self._get_stats(speaker)

        if not stats.is_known:
            return ""

        if stats.is_reliable:
            return (
                f"SPEAKER CONTEXT: {speaker} tends to use consistent/literal "
                f"interpretations. For ambiguous colors, prefer using the same "
                f"color as existing blocks. For ambiguous quantities, prefer "
                f"the smaller/matching number."
            )

        return (
            f"SPEAKER CONTEXT: {speaker} sometimes uses unexpected interpretations. "
            f"For ambiguous elements, consider both conservative and creative options."
        )
