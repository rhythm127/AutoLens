"""
test_severity.py — Unit and API tests for the Rule-Based Safety Layer (SeverityEngine).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.safety import DiagnosticContext, SeverityEngine, SeverityLevel

client = TestClient(app)


@pytest.fixture
def engine() -> SeverityEngine:
    return SeverityEngine()


class TestSeverityEngineRules:
    """Verify deterministic LOW, MEDIUM, and HIGH severity classifications."""

    def test_low_severity_for_informational_warning_under_normal_conditions(
        self, engine: SeverityEngine
    ):
        ctx = DiagnosticContext(
            warning_type="low_fuel_warning",
            light_state="continuous",
            driving_behaviour="normal",
            unusual_noise=False,
            retrieved_manual_texts=["Refuel the vehicle as soon as convenient."],
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.LOW
        assert "Monitor the warning" in result.recommended_action

    def test_medium_severity_for_steady_engine_light_normal_driving_table_2_3(
        self, engine: SeverityEngine
    ):
        """Matches Synopsis Table 2.3: steady engine malfunction light + normal driving -> MEDIUM."""
        ctx = DiagnosticContext(
            warning_type="engine_malfunction_indicator",
            light_state="continuous",
            driving_behaviour="normal",
            unusual_noise=False,
            retrieved_manual_texts=[
                "The engine or emissions system may have detected a fault."
            ],
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.MEDIUM
        assert "Arrange a professional" in result.recommended_action

    def test_high_severity_when_check_engine_light_is_flashing_section_3_1(
        self, engine: SeverityEngine
    ):
        """Matches Synopsis Section 3.1: flashing engine light -> HIGH severity."""
        ctx = DiagnosticContext(
            warning_type="engine_malfunction_indicator",
            light_state="flashing",
            driving_behaviour="shaking",
            unusual_noise=False,
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.HIGH
        assert "stop driving immediately" in result.recommended_action.lower()

    def test_high_severity_for_critical_oil_pressure_warning(
        self, engine: SeverityEngine
    ):
        ctx = DiagnosticContext(
            warning_type="oil_pressure_warning",
            light_state="continuous",
            driving_behaviour="normal",
            unusual_noise=False,
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.HIGH

    def test_high_severity_triggered_by_manual_urgent_stop_directive(
        self, engine: SeverityEngine
    ):
        """Even if a warning is unknown, 'stop the vehicle immediately' in the manual forces HIGH."""
        ctx = DiagnosticContext(
            warning_type="custom_sensor_alert",
            light_state="continuous",
            driving_behaviour="normal",
            unusual_noise=False,
            retrieved_manual_texts=[
                "Warning: Stop the vehicle immediately in a safe place and turn off the engine."
            ],
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.HIGH
        assert any("urgent stop directive" in r for r in result.triggered_rules)

    def test_high_severity_for_compound_abnormal_driving_and_noise(
        self, engine: SeverityEngine
    ):
        ctx = DiagnosticContext(
            warning_type="low_fuel_warning",
            light_state="continuous",
            driving_behaviour="stalling",
            unusual_noise=True,
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.HIGH

    def test_unknown_warning_defaults_conservatively_to_medium_never_low(
        self, engine: SeverityEngine
    ):
        ctx = DiagnosticContext(
            warning_type="unrecognized_dashboard_symbol",
            light_state="continuous",
            driving_behaviour="normal",
            unusual_noise=False,
        )
        result = engine.evaluate(ctx)
        assert result.severity == SeverityLevel.MEDIUM

    def test_invalid_inputs_raise_value_error(self, engine: SeverityEngine):
        with pytest.raises(ValueError, match="warning_type"):
            engine.evaluate(DiagnosticContext(warning_type="   "))

        with pytest.raises(ValueError, match="Invalid light_state"):
            engine.evaluate(
                DiagnosticContext(
                    warning_type="oil_pressure_warning", light_state="purple"
                )
            )


def test_diagnostics_evaluate_endpoint():
    """Verify POST /diagnostics/evaluate returns structured assessment and citations list."""
    payload = {
        "warning_type": "engine_malfunction_indicator",
        "light_state": "continuous",
        "driving_behaviour": "normal",
        "unusual_noise": False,
    }
    response = client.post("/diagnostics/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["assessment"]["severity"] == "MEDIUM"
    assert "disclaimer" in data["assessment"]
    assert isinstance(data["citations"], list)