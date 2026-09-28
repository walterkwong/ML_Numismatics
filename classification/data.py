from __future__ import annotations

import concurrent.futures as cf
import logging
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from torch.utils.data import Dataset

from image_prep import prep_reference_image

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


# Face value from category/name
CATEGORY_FACE_VALUE_RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\$2\.50"), 250.0),
    (re.compile(r"\$3\b"), 300.0),
    (re.compile(r"\$4\b"), 400.0),
    (re.compile(r"\$5\b"), 500.0),
    (re.compile(r"\$10\b"), 1000.0),
    (re.compile(r"\$20\b"), 2000.0),
    (re.compile(r"half cent", re.I), 0.5),
    (re.compile(r"two cent", re.I), 2.0),
    (re.compile(r"three dollar", re.I), 300.0),
    (re.compile(r"three cent", re.I), 3.0),
    (re.compile(r"twenty cent", re.I), 20.0),
    (re.compile(r"five cent|nickel", re.I), 5.0),
    (re.compile(r"half dime", re.I), 5.0),
    (re.compile(r"\bcent", re.I), 1.0),
    (re.compile(r"\bdime", re.I), 10.0),
    (re.compile(r"quarter eagle", re.I), 250.0),
    (re.compile(r"\bquarter", re.I), 25.0),
    (re.compile(r"half dollar", re.I), 50.0),
    (re.compile(r"half eagle", re.I), 500.0),
    (re.compile(r"double eagle", re.I), 2000.0),
    (re.compile(r"\beagle", re.I), 1000.0),
    (re.compile(r"gold dollar", re.I), 100.0),
    (re.compile(r"trade dollar|\bdollar", re.I), 100.0),
]

NAME_FACE_VALUE_RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\$10\b"), 1000.0),
    (re.compile(r"\$5\b"), 500.0),
    (re.compile(r"\$20\b"), 2000.0),
    (re.compile(r"\b50C\b", re.I), 50.0),
    (re.compile(r"\b1/2C\b", re.I), 0.5),
    (re.compile(r"\b1C\b", re.I), 1.0),
    (re.compile(r"\b1/4P\b", re.I), 0.25),
    (re.compile(r"\b1/2P\b", re.I), 0.5),
    (re.compile(r"\b1P\b", re.I), 1.0),
    (re.compile(r"\b2P\b", re.I), 2.0),
    (re.compile(r"\b3P\b", re.I), 3.0),
    (re.compile(r"\b4P\b", re.I), 4.0),
    (re.compile(r"\b6P\b", re.I), 6.0),
    (re.compile(r"\b1S\b", re.I), 12.0),  # shilling
    (re.compile(r"\b5S\b", re.I), 5.0),   # sols
    (re.compile(r"\b15S\b", re.I), 15.0),
    (re.compile(r"\b20S\b", re.I), 20.0),
    (re.compile(r"\b9D\b", re.I), 9.0),
    (re.compile(r"\b12D\b", re.I), 12.0),
    (re.compile(r"\b15D\b", re.I), 15.0),
    (re.compile(r"\b30D\b", re.I), 30.0),
    (re.compile(r"\b1/2SM\b", re.I), 0.5),
    (re.compile(r"\b1SM\b", re.I), 1.0),
    (re.compile(r"\b1/2R\b", re.I), 0.5),
    (re.compile(r"\b1/24R\b", re.I), 1.0 / 24.0),
    (re.compile(r'"4R"', re.I), 4.0),
]


def infer_face_value(category: str, name: str) -> Optional[float]:
    for pattern, value in CATEGORY_FACE_VALUE_RULES:
        if pattern.search(category):
            return value

    for pattern, value in NAME_FACE_VALUE_RULES:
        if pattern.search(name):
            return value

    return None


YEAR_RE = re.compile(r"(?:c\.?|(?:\(|^|\s))(\d{4})(?:\)|\s|$)")
RANGE_RE = re.compile(r"\((\d{4})\s*-\s*(?:(\d{4})|date)\)", re.I)

YEAR_MIN, YEAR_MAX = 1600.0, 2026.0


def extract_year(name: str, category: str) -> Optional[float]:
    match = YEAR_RE.search(name)

    if match:
        year = float(match.group(1))
        if YEAR_MIN <= year <= YEAR_MAX:
            return year

    match = RANGE_RE.search(category)

    if match:
        start = float(match.group(1))
        end = float(match.group(2)) if match.group(2) else 2025.0
        return (start + end) / 2.0

    return None


def normalize_year(year: float) -> float:
    return (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)


def denormalize_year(norm: float) -> float:
    return norm * (YEAR_MAX - YEAR_MIN) + YEAR_MIN


def sanitize_big_family(big_family: str, category: str) -> str:
    if "Nova Constellatio" in category:
        return "Nova Constellatio"
    return big_family


@dataclass
class CoinRecord:
    variety_id: int
    variety_key: str
    family_id: int
    category_id: int
    face_value_id: Optional[int]
    year: Optional[float]
    name: str
    category: str
    big_family: str
    obverse_path: Optional[Path] = None
    reverse_path: Optional[Path] = None
    seen_in_training: bool = True


@dataclass
class LabelMaps:
    family2idx: dict
    category2idx: dict
    variety2idx: dict
    face_value2idx: dict

    @property
    def num_families(self):
        return len(self.family2idx)

    @property
    def num_categories(self):
        return len(self.category2idx)

    @property
    def num_varieties(self):
        return len(self.variety2idx)

    @property
    def num_face_values(self):
        return len(self.face_value2idx)


def build_label_maps(raw_records: list[dict]) -> LabelMaps:
    families = sorted({
        sanitize_big_family(r["big_family"], r["category"])
        for r in raw_records
    })

    categories = sorted({r["category"] for r in raw_records})

    varieties = sorted({
        f'{r["category"]}::{r["name"]}'
        for r in raw_records
    })

    face_values = sorted({
        value
        for value in (
            infer_face_value(r["category"], r["name"])
            for r in raw_records
        )
        if value is not None
    })

    return LabelMaps(
        family2idx={value: i for i, value in enumerate(families)},
        category2idx={value: i for i, value in enumerate(categories)},
        variety2idx={value: i for i, value in enumerate(varieties)},
        face_value2idx={value: i for i, value in enumerate(face_values)},
    )


def _download_and_process(
    url: str,
    dest: Path,
    canvas_size: int,
    timeout: int = 20,
) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True

    try:
        response = _SESSION.get(url, timeout=timeout)
        response.raise_for_status()

        arr = np.frombuffer(response.content, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        processed = prep_reference_image(img_rgb, canvas_size)

        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(dest),
            cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
        )
        return True

    except Exception as exc:
        log.warning(f"Failed to process {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def build_records(
    raw_records: list[dict],
    img_dir: Path,
    label_maps: LabelMaps,
    canvas_size: int = 512,
    max_workers: int = 16,
) -> list[CoinRecord]:
    img_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    records = []

    for idx, r in enumerate(raw_records):
        big_family = sanitize_big_family(
            r["big_family"],
            r["category"],
        )
        variety_key = f'{r["category"]}::{r["name"]}'
        face_value = infer_face_value(r["category"], r["name"])

        record = CoinRecord(
            variety_id=label_maps.variety2idx[variety_key],
            variety_key=variety_key,
            family_id=label_maps.family2idx[big_family],
            category_id=label_maps.category2idx[r["category"]],
            face_value_id=(
                label_maps.face_value2idx[face_value]
                if face_value is not None
                else None
            ),
            year=extract_year(r["name"], r["category"]),
            name=r["name"],
            category=r["category"],
            big_family=big_family,
        )

        records.append(record)

        if r.get("obverse_image"):
            dest = img_dir / f"{record.variety_id}_obv.jpg"
            jobs.append((idx, "obverse", r["obverse_image"], dest))

        if r.get("reverse_image"):
            dest = img_dir / f"{record.variety_id}_rev.jpg"
            jobs.append((idx, "reverse", r["reverse_image"], dest))

    log.info(f"Downloading {len(jobs)} reference images...")

    obtained = {idx: {} for idx, *_ in jobs}

    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download_and_process,
                url,
                dest,
                canvas_size,
            ): (idx, face, dest)
            for idx, face, url, dest in jobs
        }

        for future in cf.as_completed(futures):
            idx, face, dest = futures[future]
            if future.result():
                obtained[idx][face] = dest

    for idx, faces in obtained.items():
        records[idx].obverse_path = faces.get("obverse")
        records[idx].reverse_path = faces.get("reverse")

    return [
        record
        for record in records
        if record.obverse_path or record.reverse_path
    ]


def save_index(
    records: list[CoinRecord],
    label_maps: LabelMaps,
    path: Path,
) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {"records": records, "label_maps": label_maps},
            f,
        )


def load_index(path: Path) -> tuple[list[CoinRecord], LabelMaps]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["records"], data["label_maps"]


def split_records(
    records: list[CoinRecord],
    heldout_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[CoinRecord], list[CoinRecord]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n_heldout = int(len(shuffled) * heldout_frac)
    heldout = shuffled[:n_heldout]
    train = shuffled[n_heldout:]

    for record in heldout:
        record.seen_in_training = False

    return train, heldout


class CoinDataset(Dataset):
    def __init__(
        self,
        records: list[CoinRecord],
        transform,
        samples_per_face: int = 1,
        faces: tuple[str, ...] = ("obverse", "reverse"),
    ):
        self.items = []

        for record in records:
            if "obverse" in faces and record.obverse_path is not None:
                self.items.append(
                    (record, record.obverse_path, "obverse")
                )

            if "reverse" in faces and record.reverse_path is not None:
                self.items.append(
                    (record, record.reverse_path, "reverse")
                )

        self.transform = transform
        self.samples_per_face = max(1, samples_per_face)

    def __len__(self):
        return len(self.items) * self.samples_per_face

    def __getitem__(self, idx):
        record, path, face = self.items[idx % len(self.items)]

        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img_rgb)["image"]

        year = (
            normalize_year(record.year)
            if record.year is not None
            else float("nan")
        )

        target = {
            "family": record.family_id,
            "category": record.category_id,
            "variety": record.variety_id,
            "face_value": (
                record.face_value_id
                if record.face_value_id is not None
                else -1
            ),
            "year": year,
            "seen_in_training": record.seen_in_training,
        }

        return image, target