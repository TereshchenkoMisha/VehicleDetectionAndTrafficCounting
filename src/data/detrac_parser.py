import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .config import RAW_ROOT


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    def as_xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class Target:
    track_id: int
    cls: str
    box: Box
    truncated: float
    occlusion: float


@dataclass
class FrameAnn:
    frame_id: int
    targets: list


@dataclass
class SequenceAnn:
    name: str
    frames: dict = field(default_factory=dict)
    ignored_regions: list = field(default_factory=list)


@dataclass
class SequenceMeta:
    name: str
    weather: str = "unknown"
    scene: str = "unknown"


def _parse_box(el) -> Box:
    if el.get("left") is not None:
        return Box(float(el.get("left")), float(el.get("top")),
                   float(el.get("width")), float(el.get("height")))
    child = el.find("box")
    if child is not None and child.get("left") is not None:
        return Box(float(child.get("left")), float(child.get("top")),
                   float(child.get("width")), float(child.get("height")))
    s = (el.get("box") or "").strip("[]() ")
    x, y, w, h = (float(v) for v in s.replace(",", " ").split())
    return Box(x, y, w, h)


def _target_class(t) -> str:
    v = t.get("vehicle_type")
    if v:
        return v.strip().lower()
    attr = t.find("attribute")
    if attr is not None:
        v = attr.get("vehicle_type")
        if v:
            return v.strip().lower()
    raise ValueError(f"<target> without vehicle_type: {t.attrib}")


def _target_truncation(t) -> float:
    v = t.get("truncated")
    if v is not None:
        return float(v)
    attr = t.find("attribute")
    if attr is not None and attr.get("truncation_ratio") is not None:
        return float(attr.get("truncation_ratio"))
    return 0.0


def _parse_ann_root(root) -> SequenceAnn:
    seq = SequenceAnn(name=root.get("name") or "unknown")
    for region in root.iter("ignored_region"):
        for b in region.findall("box"):
            seq.ignored_regions.append(_parse_box(b))
    for frame in root.iter("frame"):
        fid_raw = frame.get("num") or frame.get("frame_id")
        fid = int(fid_raw)
        tl = frame.find("target_list")
        targets = []
        if tl is not None:
            for t in tl.findall("target"):
                targets.append(Target(int(t.get("id")), _target_class(t),
                                      _parse_box(t), _target_truncation(t), 0.0))
        seq.frames[fid] = FrameAnn(fid, targets)
    return seq


def _parse_meta_root(root) -> SequenceMeta:
    def text(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None
    return SequenceMeta(root.get("name") or "unknown",
                        (text("weather") or "unknown").lower(),
                        (text("scene") or "unknown").lower())


def load_all(root=RAW_ROOT):
    anns, metas = {}, {}
    for p in tqdm(sorted(Path(root).rglob("MVI_*.xml")), desc="parse XML", unit="file"):
        r = ET.parse(p).getroot()
        if r.find("weather") is not None:
            m = _parse_meta_root(r)
            metas[m.name] = m
        elif r.find("frame") is not None:
            a = _parse_ann_root(r)
            anns[a.name] = a
    return anns, metas


def find_sequence_dir(frames_root, name):
    for cand in Path(frames_root).rglob(name):
        if cand.is_dir() and any(cand.glob("*.jpg")):
            return cand
    return None