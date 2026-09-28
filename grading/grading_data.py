from __future__ import annotations

import logging
import pickle
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from handcrafted_features import extract_handcrafted_features
from torch.utils.data import Dataset


log = logging.getLogger(__name__)


# Sheldon grade
ADJECTIVAL_TIERS = [
    "PO_FR_AG",  # 1-3
    "G",         # 4-6
    "VG",        # 8-10
    "F",         # 12-15
    "VF",        # 20-35
    "EF",        # 40-45
    "AU",        # 50-58
    "MS_LOW",    # 60-62
    "MS_MID",    # 63-64
    "MS_GEM",    # 65-70
]


def grade_to_tier_id(grade: float) -> int:
    if grade < 4:
        return 0
    if grade < 8:
        return 1
    if grade < 12:
        return 2
    if grade < 20:
        return 3
    if grade < 40:
        return 4
    if grade < 50:
        return 5
    if grade < 60:
        return 6
    if grade < 63:
        return 7
    if grade < 65:
        return 8
    return 9


def grade_to_state_id(grade: float) -> int:
    return int(grade >= 60)


@dataclass
class GradeRecord:
    row_id: int
    coin_id: str
    grade: float
    tier_id: int
    state_id: int
    front_path: Path
    back_path: Path
    source_url: str


def load_grade_records(
    db_path: Path,
    base_dir: Path | None = None,
    table: str = "coins",
) -> list[GradeRecord]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()

    records = []
    skipped = 0

    for row in rows:
        front = Path(row["front_local_path"])
        back = Path(row["back_local_path"])

        if base_dir and not front.is_absolute():
            front = base_dir / front
        if base_dir and not back.is_absolute():
            back = base_dir / back

        if not front.exists() or not back.exists():
            skipped += 1
            continue

        grade = float(row["grade"])

        records.append(
            GradeRecord(
                row_id=row["id"],
                coin_id=row["coin_id"],
                grade=grade,
                tier_id=grade_to_tier_id(grade),
                state_id=grade_to_state_id(grade),
                front_path=front,
                back_path=back,
                source_url=row["source_url"],
            )
        )

    log.info(
        f"Loaded {len(records)} grade records from {db_path} "
        f"({skipped} skipped)"
    )
    return records


def split_by_coin_id(
    records: list[GradeRecord],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[GradeRecord], list[GradeRecord]]:
    coin_ids = list({r.coin_id for r in records})
    random.Random(seed).shuffle(coin_ids)

    n_val = max(1, int(len(coin_ids) * val_frac))
    val_ids = set(coin_ids[:n_val])

    train = [r for r in records if r.coin_id not in val_ids]
    val = [r for r in records if r.coin_id in val_ids]

    return train, val


def save_split(
    train: list[GradeRecord],
    val: list[GradeRecord],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({"train": train, "val": val}, f)

    log.info(f"Saved split: {len(train)} train / {len(val)} val")


def load_split(
    path: Path,
) -> tuple[list[GradeRecord], list[GradeRecord]]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["train"], data["val"]


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _face_path(record: GradeRecord, face: int) -> Path:
    return record.front_path if face == 0 else record.back_path


def _targets(record: GradeRecord) -> dict[str, torch.Tensor]:
    return {
        "grade": torch.tensor(record.grade, dtype=torch.float32),
        "tier": torch.tensor(record.tier_id, dtype=torch.long),
        "state": torch.tensor(record.state_id, dtype=torch.long),
    }


class GradeImageDataset(Dataset):
    def __init__(
        self,
        records: list[GradeRecord],
        transform,
        samples_per_record: int = 4,
    ):
        self.records = records
        self.transform = transform
        self.samples_per_record = max(1, samples_per_record)

    def __len__(self):
        return len(self.records) * 2 * self.samples_per_record

    def __getitem__(self, idx):
        face_idx = idx % (len(self.records) * 2)
        record = self.records[face_idx % len(self.records)]
        face = face_idx // len(self.records)

        image = _load_rgb(_face_path(record, face))
        image = self.transform(image=image)["image"]

        return image, _targets(record)


class GradeFeatureDataset(Dataset):
    def __init__(
        self,
        records: list[GradeRecord],
        transform,
        samples_per_record: int = 4,
        n_color_clusters: int = 5,
        radius_frac: float = 0.48,
    ):
        self.records = records
        self.transform = transform
        self.samples_per_record = max(1, samples_per_record)
        self.n_color_clusters = n_color_clusters
        self.radius_frac = radius_frac

    def __len__(self):
        return len(self.records) * 2 * self.samples_per_record

    def __getitem__(self, idx):
        face_idx = idx % (len(self.records) * 2)
        record = self.records[face_idx % len(self.records)]
        face = face_idx // len(self.records)

        image = _load_rgb(_face_path(record, face))
        image = self.transform(image=image)["image"]
        features = extract_handcrafted_features(
            image,
            self.n_color_clusters,
            self.radius_frac,
        )

        return torch.from_numpy(features), _targets(record)