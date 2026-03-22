from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputValidator:
    required_keys: tuple[str, ...] = (
        "action",
        "answer",
        "confidence",
        "epistemic_status",
        "ask_for_clarification",
    )

    def validate(self, payload: dict) -> list[str]:
        errors: list[str] = []

        for key in self.required_keys:
            if key not in payload:
                errors.append(f"missing_key:{key}")

        confidence = payload.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                errors.append("invalid_confidence_type")
            elif not (0 <= confidence <= 1):
                errors.append("confidence_out_of_range")

        answer = payload.get("answer")
        if answer is not None and not isinstance(answer, str):
            errors.append("invalid_answer_type")

        return errors
