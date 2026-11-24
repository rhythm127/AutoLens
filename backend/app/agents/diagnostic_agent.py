"""
diagnostic_agent.py - Stateful Conversational Diagnostic Agent for AutoLens.

Fulfils Synopsis Section 2.4:
- Maintains conversation state per session (vehicle, warning_type, light_state,
  driving_behaviour, unusual_noise).
- Dynamically determines which slot is still unknown (None) and asks only the
  next relevant follow-up question.
- Parses user replies into canonical slot values.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """Tracks known and unknown diagnostic slots across chat turns."""

    session_id: str
    warning_type: str
    manufacturer: str | None = None
    model: str | None = None
    year: int | None = None
    light_state: str | None = None          # 'continuous' or 'flashing'
    driving_behaviour: str | None = None    # 'normal' or abnormal symptom
    unusual_noise: bool | None = None       # True or False
    history: list[dict[str, str]] = field(default_factory=list)

    def missing_fields(self) -> list[str]:
        """Return ordered list of diagnostic slots that are still unknown."""
        missing: list[str] = []
        if self.light_state is None:
            missing.append("light_state")
        if self.driving_behaviour is None:
            missing.append("driving_behaviour")
        if self.unusual_noise is None:
            missing.append("unusual_noise")
        return missing

    @property
    def is_complete(self) -> bool:
        """True when all 3 symptom slots have been collected."""
        return len(self.missing_fields()) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "warning_type": self.warning_type,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "year": self.year,
            "light_state": self.light_state,
            "driving_behaviour": self.driving_behaviour,
            "unusual_noise": self.unusual_noise,
            "missing_fields": self.missing_fields(),
            "is_complete": self.is_complete,
        }


FOLLOW_UP_QUESTIONS: dict[str, str] = {
    "light_state": (
        "Is the warning light continuously illuminated (steady) or flashing/blinking?"
    ),
    "driving_behaviour": (
        "How is the vehicle driving right now? (e.g., 'normal', or experiencing "
        "'shaking', 'loss of power', 'stalling', or 'pulling')"
    ),
    "unusual_noise": (
        "Are you hearing any unusual mechanical or engine noises such as knocking, "
        "grinding, or squealing? (yes/no)"
    ),
}


class DiagnosticAgent:
    """
    Stateful agent that manages diagnostic sessions and selects follow-up questions.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def start_session(
        self,
        warning_type: str,
        manufacturer: str | None = None,
        model: str | None = None,
        year: int | None = None,
        light_state: str | None = None,
        driving_behaviour: str | None = None,
        unusual_noise: bool | None = None,
    ) -> ConversationState:
        if not isinstance(warning_type, str) or not warning_type.strip():
            raise ValueError("warning_type must be a non-empty string.")

        session_id = uuid.uuid4().hex
        state = ConversationState(
            session_id=session_id,
            warning_type=warning_type.strip().lower().replace(" ", "_").replace("-", "_"),
            manufacturer=manufacturer.strip() if manufacturer else None,
            model=model.strip() if model else None,
            year=year,
            light_state=self._parse_light_state(light_state) if light_state else None,
            driving_behaviour=(
                self._parse_driving_behaviour(driving_behaviour)
                if driving_behaviour
                else None
            ),
            unusual_noise=unusual_noise,
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> ConversationState:
        if session_id not in self._sessions:
            raise KeyError(f"Diagnostic session '{session_id}' not found.")
        return self._sessions[session_id]

    def get_next_question(self, state: ConversationState) -> str | None:
        """Return the next follow-up question for the first missing slot, or None if complete."""
        missing = state.missing_fields()
        if not missing:
            return None
        return FOLLOW_UP_QUESTIONS[missing[0]]

    def process_answer(self, session_id: str, user_message: str) -> ConversationState:
        """
        Parse the user's answer for the currently active missing slot and advance state.
        """
        state = self.get_session(session_id)
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string.")

        missing = state.missing_fields()
        if not missing:
            return state

        current_slot = missing[0]
        cleaned = user_message.strip()
        state.history.append({"slot": current_slot, "user_message": cleaned})

        if current_slot == "light_state":
            state.light_state = self._parse_light_state(cleaned)
        elif current_slot == "driving_behaviour":
            state.driving_behaviour = self._parse_driving_behaviour(cleaned)
        elif current_slot == "unusual_noise":
            state.unusual_noise = self._parse_unusual_noise(cleaned)

        return state

    @staticmethod
    def _parse_light_state(text: str) -> str:
        t = text.strip().lower()
        if any(w in t for w in ("flash", "blink", "flicker")):
            return "flashing"
        if any(w in t for w in ("continuous", "steady", "solid", "constant", "stay")):
            return "continuous"
        raise ValueError(
            "Could not determine light state. Please specify 'continuous' (steady) or 'flashing' (blinking)."
        )

    @staticmethod
    def _parse_driving_behaviour(text: str) -> str:
        t = text.strip().lower()
        abnormal_keywords = {
            "shak": "shaking",
            "vibrat": "shaking",
            "stall": "stalling",
            "die": "stalling",
            "power": "loss_of_power",
            "slow": "loss_of_power",
            "sluggish": "loss_of_power",
            "pull": "pulling",
            "jerk": "jerking",
            "rough": "rough_running",
        }
        for kw, canonical in abnormal_keywords.items():
            if kw in t:
                return canonical
        if any(w in t for w in ("normal", "fine", "good", "okay", "ok", "smooth", "no issue")):
            return "normal"
        # Preserve any custom abnormal symptom description
        return t.replace(" ", "_")

    @staticmethod
    def _parse_unusual_noise(text: str) -> bool:
        t = text.strip().lower()
        if any(
            w in t
            for w in ("yes", "yeah", "yep", "true", "knock", "grind", "squeal", "rattle", "hiss", "loud", "noise")
        ) and not any(neg in t for neg in ("no ", "no,", "not ", "none", "quiet", "silent")):
            return True
        if any(w in t for w in ("no", "nope", "false", "none", "quiet", "normal", "silent", "nothing")):
            return False
        raise ValueError(
            "Could not determine if there is unusual noise. Please answer 'yes' or 'no'."
        )