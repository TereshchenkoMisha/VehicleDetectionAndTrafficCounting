"""Parse the UA-DETRAC XML annotation files into plain dataclasses.

The dataset ships one XML per sequence. The mirror we use has more than one XML layout (attributes
on the element, a child element, or a string), so the box and attribute readers accept all of them.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..config import RAW_ROOT


@dataclass
class Box:
    """Axis-aligned box: top-left corner (x, y) and size (w, h), in pixels."""
    x: float
    y: float
    w: float
    h: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Corners as x1, y1, x2, y2."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Target:
    """One annotated vehicle in one frame.

    `cls` is the raw DETRAC vehicle type (car, van, bus, others); `track_id` is stable across the
    frames of a sequence. `truncated` is read from the XML but unused. `occlusion` is not read: it
    is always 0.0.
    """
    track_id: int
    cls: str
    box: Box
    truncated: float
    occlusion: float


@dataclass
class FrameAnn:
    """All targets annotated in one frame."""
    frame_id: int
    targets: list[Target]


@dataclass
class SequenceAnn:
    """Annotations of one sequence: targets per frame, and the ignored regions.

    `frames` is keyed by the XML frame number, which can differ by one from the JPG file number
    (see `sequences.resolve_naming`). Ignored regions are the same for the whole sequence.
    """
    name: str
    frames: dict[int, FrameAnn] = field(default_factory=dict)
    ignored_regions: list[Box] = field(default_factory=list)


@dataclass
class SequenceMeta:
    """Weather and scene labels of a sequence; "unknown" when the dataset copy has none."""
    name: str
    weather: str = "unknown"
    scene: str = "unknown"


def _parse_box(el: ET.Element) -> Box:
    """Box from an XML element.

    Three layouts exist in the wild: `left/top/width/height` attributes on the element itself,
    the same attributes on a `<box>` child, or a single `box="x y w h"` string attribute.
    """
    if el.get("left") is not None:
        return Box(float(el.get("left")), float(el.get("top")),
                   float(el.get("width")), float(el.get("height")))

    child = el.find("box")
    if child is not None and child.get("left") is not None:
        return Box(float(child.get("left")), float(child.get("top")),
                   float(child.get("width")), float(child.get("height")))

    text = (el.get("box") or "").strip("[]() ")
    x, y, w, h = (float(v) for v in text.replace(",", " ").split())
    return Box(x, y, w, h)


def _target_class(target: ET.Element) -> str:
    """Vehicle type of a <target>, lower-case. It sits on the element or on its <attribute> child."""
    value = target.get("vehicle_type")
    if value:
        return value.strip().lower()
    attr = target.find("attribute")
    if attr is not None and attr.get("vehicle_type"):
        return attr.get("vehicle_type").strip().lower()
    raise ValueError(f"<target> without vehicle_type: {target.attrib}")


def _target_truncation(target: ET.Element) -> float:
    """Truncation ratio from the element or its <attribute> child; 0 when absent."""
    value = target.get("truncated")
    if value is not None:
        return float(value)
    attr = target.find("attribute")
    if attr is not None and attr.get("truncation_ratio") is not None:
        return float(attr.get("truncation_ratio"))
    return 0.0


def _parse_ann_root(root: ET.Element) -> SequenceAnn:
    """Annotation XML (ignored regions and per-frame target lists) -> SequenceAnn."""
    seq = SequenceAnn(name=root.get("name") or "unknown")

    # Ignored regions are listed once for the whole sequence
    for region in root.iter("ignored_region"):
        for box in region.findall("box"):
            seq.ignored_regions.append(_parse_box(box))

    # One <frame> per annotated frame, each with a <target_list>. A frame without a list has no targets.
    for frame in root.iter("frame"):
        fid = int(frame.get("num") or frame.get("frame_id"))
        target_list = frame.find("target_list")
        targets = []
        if target_list is not None:
            for t in target_list.findall("target"):
                targets.append(Target(int(t.get("id")), _target_class(t),
                                      _parse_box(t), _target_truncation(t), 0.0))
        seq.frames[fid] = FrameAnn(fid, targets)
    return seq


def _parse_meta_root(root: ET.Element) -> SequenceMeta:
    """Metadata XML (<weather>, <scene>) -> SequenceMeta."""
    def text(tag: str) -> str | None:
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    return SequenceMeta(root.get("name") or "unknown",
                        (text("weather") or "unknown").lower(),
                        (text("scene") or "unknown").lower())


def load_all(root: str | Path = RAW_ROOT) -> tuple[dict[str, SequenceAnn], dict[str, SequenceMeta]]:
    """Parse every `MVI_*.xml` under `root`.

    The two kinds of file are told apart by content: a file with a <weather> element is metadata,
    a file with <frame> elements is an annotation. Returns (annotations, metadata), both keyed by
    sequence name. The metadata dict is empty when the dataset copy has no metadata files.
    """
    anns: dict[str, SequenceAnn] = {}
    metas: dict[str, SequenceMeta] = {}
    for path in tqdm(sorted(Path(root).rglob("MVI_*.xml")), desc="parse XML", unit="file"):
        xml_root = ET.parse(path).getroot()
        if xml_root.find("weather") is not None:
            meta = _parse_meta_root(xml_root)
            metas[meta.name] = meta
        elif xml_root.find("frame") is not None:
            ann = _parse_ann_root(xml_root)
            anns[ann.name] = ann
    return anns, metas
