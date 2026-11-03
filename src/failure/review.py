"""Manual review of failure cases: check what the reviewer wrote against the saved events, and write the report.

The automatic analysis only finds candidates. A case becomes a result when a person has looked at the
frames and written down what failed, the likely cause and its limits (reports/failure_review.json):

    {"cases": [{"case_id": ..., "label": ..., "observed_failure": ..., "likely_cause": ..., "cause_confidence": ...,
                "counting_impact": ..., "limitations": ..., "possible_improvement": ..., "reference_gt_id": null,
                "counterpart_assessment": {"<configuration>": {"counted_track_ids": [...], "counted_events": 1,
                                                               "counted_classes": ["car"], "description": ...}}}]}

`counterpart_assessment` says how every configuration counted the same vehicle. It is a statement of
fact, so it is checked against the saved crossing events: a review that contradicts them is rejected.
"""
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import CLASSES
from .diagnosis import FailureRow, event_of, find_case, vehicle_key

REQUIRED_FIELDS: tuple[str, ...] = ("observed_failure", "likely_cause", "cause_confidence", "counting_impact",
                                    "limitations")
OPTIONAL_FIELDS: tuple[str, ...] = ("possible_improvement",)
MIN_VEHICLES: int = 3       # the course asks for at least three failure cases


def check_assessment(case: dict[str, Any], row: FailureRow,
                     predictions: dict[str, dict[str, Any]]) -> None:
    """Raise ValueError unless the reviewer's account of every configuration matches the saved events.

    `predictions` is the `count_tracks` result of one split: {configuration: {sequence: {"events": ...}}}.
    The events of the listed tracker ids must have the stated number and classes.
    """
    for config, stated in case["counterpart_assessment"].items():
        events = predictions[config][row["sequence"]]["events"]
        counted = [event for event in events if event[1] in stated["counted_track_ids"]]
        classes = sorted(CLASSES[event[2]] for event in counted)
        if len(counted) != stated["counted_events"] or classes != sorted(stated["counted_classes"]):
            raise ValueError(f"{case['case_id']}: the assessment of {config} contradicts the saved crossing events")


def reviewed_vehicle(case: dict[str, Any], row: FailureRow) -> tuple[str, int, bool]:
    """Vehicle behind a reviewed case; the reviewer's `reference_gt_id` names the GT vehicle of an extra crossing."""
    reference = case.get("reference_gt_id")
    return (row["sequence"], reference, True) if reference is not None else vehicle_key(row)


def check_review(cases: Sequence[dict[str, Any]], rows: Sequence[FailureRow],
                 predictions: dict[str, dict[str, Any]]) -> list[FailureRow]:
    """Validate every reviewed case and return its row; ValueError on the first problem.

    A case needs an existing id, every text field filled in, an assessment that matches the saved events,
    and the cases together must show at least MIN_VEHICLES different vehicles.
    """
    selected = []
    for case in cases:
        row = find_case(rows, case["case_id"])
        missing = [field for field in REQUIRED_FIELDS if not case.get(field, "").strip()]
        if missing:
            raise ValueError(f"{case['case_id']}: empty review fields {missing}")
        check_assessment(case, row, predictions)
        selected.append(row)

    # Two cases of the same vehicle in different configurations are still one failure case
    vehicles = {reviewed_vehicle(case, row) for case, row in zip(cases, selected, strict=True)}
    if len(vehicles) < MIN_VEHICLES:
        raise ValueError(f"Need at least {MIN_VEHICLES} different vehicles, the review has {len(vehicles)}")
    return selected


def markdown_report(cases: Sequence[dict[str, Any]], rows: Sequence[FailureRow], picture_dir: Path,
                    out_path: Path) -> str:
    """Markdown with one section per reviewed case: the comparison picture, the account and the limits.

    Pictures are linked relative to `out_path`, so the report can be moved together with them.
    """
    md = ["# Reviewed failure cases", "",
          "Each case compares the four configurations on the same vehicle and frames. The cases explain "
          "observed behaviour; they do not show how often each cause occurs.", ""]
    for case in cases:
        row = find_case(rows, case["case_id"])
        picture = Path(os.path.relpath(picture_dir / case["case_id"] / "comparison.png", out_path.parent))
        md += [f"## {case.get('label', case['case_id'])}", "",
               f"`{case['case_id']}` - sequence {row['sequence']}, frame {event_of(row)['frame']}", "",
               f"![comparison]({picture.as_posix()})", ""]
        md += [f"**{field.replace('_', ' ').capitalize()}**: {case[field]}  "
               for field in REQUIRED_FIELDS + OPTIONAL_FIELDS if case.get(field)]
        md += ["", "**Same vehicle in every configuration**", ""]
        md += [f"- {config}: {stated['description']} ({stated['counted_events']} counted)"
               for config, stated in case["counterpart_assessment"].items()]
        md.append("")
    return "\n".join(md) + "\n"

