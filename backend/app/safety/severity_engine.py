"""
severity_engine.py — Rule-Based Safety Layer & Severity Engine for AutoLens.

As specified in the AutoLens architecture (Section 2.4):
- The language model is NEVER allowed to invent or freely decide severity.
- Severity (LOW, MEDIUM, HIGH) is determined strictly by deterministic rules
  combining:
    1. Canonical warning category (safety-critical vs. standard vs. informational)
    2. Explicit urgency markers found in the retrieved vehicle manual text
    3. Driver-reported context (light_state, driving_behaviour, unusual_noise)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SeverityLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Canonical warning categories
CRITICAL_WARNINGS: set[str] = {
    "oil_pressure_warning",
    "brake_system_warning",
    "engine_temperature_warning",
    "coolant_temperature_warning",
    "charging_system_warning",
    "battery_charge_warning",
    "airbag_srs_warning",
    "power_steering_warning",
}

STANDARD_WARNINGS: set[str] = {
    "engine_malfunction_indicator",
    "check_engine_light",
    "tire_pressure_warning",
    "tpms_warning",
    "abs_warning",
    "traction_control_warning",
    "transmission_warning",
}

INFORMATIONAL_WARNINGS: set[str] = {
    "low_fuel_warning",
    "washer_fluid_low",
    "maintenance_required",
    "door_ajar_warning",
    "seatbelt_reminder",
}

# Phrases in the retrieved owner's manual that mandate immediate HIGH severity
URGENT_MANUAL_PHRASES: tuple[str, ...] = (
    "stop the vehicle immediately",
    "pull over immediately",
    "do not continue driving",
    "turn off the engine immediately",
    "switch off the engine",
    "catastrophic damage",
    "dangerously low",
)

RECOMMENDED_ACTIONS: dict[SeverityLevel, str] = {
    SeverityLevel.LOW: (
        "Monitor the warning light. Consult your vehicle owner's manual or schedule "
        "a routine check at an authorized service center if the indicator persists."
    ),
    SeverityLevel.MEDIUM: (
        "Avoid unnecessary or high-speed driving. Arrange a professional diagnostic "
        "inspection at an authorized service center as soon as possible."
    ),
    SeverityLevel.HIGH: (
        "Pull over safely and stop driving immediately. Switch off the engine and "
        "contact roadside assistance or an authorized mechanic."
    ),
}

SAFETY_DISCLAIMER: str = (
    "AutoLens provides troubleshooting guidance grounded in your vehicle's manual "
    "and does not replace a professional mechanical inspection."
)


@dataclass(frozen=True)
class DiagnosticContext:
    """
    Structured state collected from the Vision Agent, RAG Retriever, and User Chat.

    Attributes
    ----------
    warning_type : str
        Normalized warning identifier (e.g., 'engine_malfunction_indicator').
    light_state : str
        'continuous' (steady) or 'flashing' (blinking).
    driving_behaviour : str
        'normal' or abnormal descriptions ('shaking', 'loss_of_power', 'stalling', 'pulling').
    unusual_noise : bool
        True if the driver hears knocking, grinding, squealing, or other abnormal sounds.
    retrieved_manual_texts : list[str]
        Raw text chunks retrieved from the vehicle's manual via HybridRetriever.
    """

    warning_type: str
    light_state: str = "continuous"
    driving_behaviour: str = "normal"
    unusual_noise: bool = False
    retrieved_manual_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeverityAssessment:
    """
    Deterministic output from the Rule-Based Safety Layer.
    """

    severity: SeverityLevel
    recommended_action: str
    triggered_rules: list[str]
    disclaimer: str = SAFETY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "recommended_action": self.recommended_action,
            "triggered_rules": list(self.triggered_rules),
            "disclaimer": self.disclaimer,
        }


class SeverityEngine:
    """
    Evaluates DiagnosticContext using explicit, deterministic safety rules.
    """

    @staticmethod
    def _normalize_warning(warning_type: str) -> str:
        if not isinstance(warning_type, str) or not warning_type.strip():
            raise ValueError("warning_type must be a non-empty string.")
        return warning_type.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _normalize_light_state(light_state: str) -> str:
        state = light_state.strip().lower() if isinstance(light_state, str) else ""
        if state not in ("continuous", "steady", "flashing", "blinking"):
            raise ValueError(
                f"Invalid light_state '{light_state}'. Must be 'continuous' or 'flashing'."
            )
        return "flashing" if state in ("flashing", "blinking") else "continuous"

    @staticmethod
    def _normalize_driving_behaviour(behaviour: str) -> str:
        if not isinstance(behaviour, str) or not behaviour.strip():
            raise ValueError("driving_behaviour must be a non-empty string.")
        return behaviour.strip().lower()

    @classmethod
    def _manual_contains_urgent_directive(cls, texts: list[str]) -> str | None:
        """Return the matched urgent phrase if any retrieved chunk orders an immediate stop."""
        combined = " ".join(t.lower() for t in texts if isinstance(t, str))
        for phrase in URGENT_MANUAL_PHRASES:
            if phrase in combined:
                return phrase
        return None

    def evaluate(self, context: DiagnosticContext) -> SeverityAssessment:
        """
        Apply safety rules in strict priority order (HIGH -> MEDIUM -> LOW).
        """
        warning = self._normalize_warning(context.warning_type)
        light_state = self._normalize_light_state(context.light_state)
        behaviour = self._normalize_driving_behaviour(context.driving_behaviour)
        if not isinstance(context.unusual_noise, bool):
            raise ValueError("unusual_noise must be a boolean.")

        is_abnormal_driving = behaviour != "normal"
        urgent_phrase = self._manual_contains_urgent_directive(
            context.retrieved_manual_texts
        )

        triggered_high: list[str] = []
        triggered_medium: list[str] = []

        # --- HIGH SEVERITY RULES (Any match immediately escalates to HIGH) ---
        if warning in CRITICAL_WARNINGS:
            triggered_high.append(
                f"Critical safety warning indicator detected: '{warning}'."
            )

        if urgent_phrase is not None:
            triggered_high.append(
                f"Vehicle manual contains urgent stop directive: '{urgent_phrase}'."
            )

        if light_state == "flashing" and warning not in INFORMATIONAL_WARNINGS:
            triggered_high.append(
                f"Warning indicator '{warning}' is flashing/blinking, indicating an active fault."
            )

        if is_abnormal_driving and context.unusual_noise:
            triggered_high.append(
                f"Compound mechanical distress: abnormal driving ('{behaviour}') combined with unusual noise."
            )

        if triggered_high:
            return SeverityAssessment(
                severity=SeverityLevel.HIGH,
                recommended_action=RECOMMENDED_ACTIONS[SeverityLevel.HIGH],
                triggered_rules=triggered_high,
            )

        # --- MEDIUM SEVERITY RULES ---
        if is_abnormal_driving:
            triggered_medium.append(
                f"Vehicle exhibits abnormal driving behaviour: '{behaviour}'."
            )

        if context.unusual_noise:
            triggered_medium.append(
                "Driver reported unusual mechanical or engine noise."
            )

        if light_state == "flashing" and warning in INFORMATIONAL_WARNINGS:
            triggered_medium.append(
                f"Informational indicator '{warning}' is flashing abnormally."
            )

        if warning in STANDARD_WARNINGS:
            triggered_medium.append(
                f"Standard fault indicator '{warning}' is continuously illuminated."
            )

        if warning not in INFORMATIONAL_WARNINGS and warning not in STANDARD_WARNINGS:
            # Unknown / uncatalogued warning defaults conservatively to MEDIUM, never LOW
            triggered_medium.append(
                f"Uncatalogued warning '{warning}' defaults conservatively to MEDIUM severity."
            )

        if triggered_medium:
            return SeverityAssessment(
                severity=SeverityLevel.MEDIUM,
                recommended_action=RECOMMENDED_ACTIONS[SeverityLevel.MEDIUM],
                triggered_rules=triggered_medium,
            )

        # --- LOW SEVERITY RULES ---
        return SeverityAssessment(
            severity=SeverityLevel.LOW,
            recommended_action=RECOMMENDED_ACTIONS[SeverityLevel.LOW],
            triggered_rules=[
                f"Non-urgent indicator '{warning}' with normal driving behaviour and no abnormal noise."
            ],
        )