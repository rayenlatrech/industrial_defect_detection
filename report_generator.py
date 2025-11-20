from dataclasses import dataclass
from typing import Optional


# ----------------------------------------------------------------------
# Data structure for inspection info
# ----------------------------------------------------------------------


@dataclass
class InspectionInput:
    """
    Structured information about a single inspected part.
    This is what will be passed to the report generator.
    """
    image_id: str                # e.g. filename or index
    predicted_class: str         # "ok_front" or "def_front"
    predicted_prob: float        # probability of predicted class (0..1)
    anomaly_score: float         # e.g. kNN distance or SVM score
    anomaly_model: str = "kNN"   # which anomaly model produced the score
    additional_comment: Optional[str] = None  # e.g. "Grad-CAM highlights top-left area"


# ----------------------------------------------------------------------
# Helper functions for rule-based reporting
# ----------------------------------------------------------------------


def _format_confidence(prob: float) -> str:
    """
    Turn a probability (0..1) into a human-readable confidence label.
    """
    if prob >= 0.98:
        return "very high"
    elif prob >= 0.95:
        return "high"
    elif prob >= 0.90:
        return "moderate"
    else:
        return "low"


def _risk_level_from_anomaly_score(score: float) -> str:
    """
    Simple heuristic to turn an anomaly score into a qualitative risk level.
    Thresholds are tuned for kNN distance: normal scores small, defects large.
    Adjust if your anomaly distribution changes.
    """
    if score < 20:
        return "low"
    elif score < 60:
        return "medium"
    else:
        return "high"


def generate_report_rule_based(info: InspectionInput) -> str:
    """
    Rule-based report generator (no LLM required).
    Turns structured info into a readable inspection report.
    """
    cls = info.predicted_class
    prob = info.predicted_prob
    anomaly = info.anomaly_score

    confidence_label = _format_confidence(prob)
    risk_level = _risk_level_from_anomaly_score(anomaly)

    if cls == "ok_front":
        status_line = "Overall assessment: PART ACCEPTED (no clear defect detected)."
        detail_line = (
            "The classifier did not detect any visible surface defects on the casting. "
            "The anomaly score is consistent with previously observed normal parts."
        )
    else:
        status_line = "Overall assessment: PART REJECTED (surface defect suspected)."
        detail_line = (
            "The classifier detected visual patterns consistent with known surface defects. "
            "The anomaly score indicates that this part deviates significantly from the distribution "
            "of normal (OK) castings."
        )

    comment_block = ""
    if info.additional_comment:
        comment_block = f"\n\nAdditional analysis:\n- {info.additional_comment}"

    report = f"""Industrial Casting Quality Inspection Report
--------------------------------------------------
Part identifier: {info.image_id}

Predicted class: {cls}
Model confidence: {prob:.3f} ({confidence_label} confidence)
Anomaly model: {info.anomaly_model}
Anomaly score: {anomaly:.3f} (estimated risk level: {risk_level})

{status_line}

Details:
- {detail_line}
- This decision is based on a convolutional neural network trained on labeled casting images,
  combined with an anomaly detection model trained on defect-free parts.

Recommendations:
- If the part is marked as REJECTED:
  • Remove the part from the production line for manual inspection.
  • Inspect the mold and process parameters for possible sources of porosity or surface irregularities.
- If the part is marked as ACCEPTED:
  • No immediate action required, but continue periodic sampling and monitoring.

Note:
- This automated assessment should be used as a decision-support tool and not as the sole
  criterion for critical safety decisions.{comment_block}
"""
    return report
