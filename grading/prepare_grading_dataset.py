from __future__ import annotations

import logging
from pathlib import Path

import cv2
import grading_config
from grading_data import (
    GradeRecord,
    load_grade_records,
    save_split,
    split_by_coin_id,
)
from image_prep import prep_reference_image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _prep_and_save_face(
    src_path: Path,
    dst_path: Path,
    canvas_size: int,
) -> bool:
    if dst_path.exists():
        return True

    image = cv2.imread(str(src_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = prep_reference_image(
        image,
        canvas_size=canvas_size,
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst_path), image)


def prep_all_records_offline(
    records: list[GradeRecord],
    out_dir: Path,
    canvas_size: int,
) -> list[GradeRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepped_records = []

    log.info(f"Preparing {len(records)} records...")

    for record in tqdm(records, desc="Preparing images"):
        front_dst = out_dir / f"{record.coin_id}_{record.row_id}_front.jpg"
        back_dst = out_dir / f"{record.coin_id}_{record.row_id}_back.jpg"

        front_ok = _prep_and_save_face(
            record.front_path,
            front_dst,
            canvas_size,
        )
        back_ok = _prep_and_save_face(
            record.back_path,
            back_dst,
            canvas_size,
        )

        if not (front_ok and back_ok):
            continue

        prepped_records.append(
            GradeRecord(
                row_id=record.row_id,
                coin_id=record.coin_id,
                grade=record.grade,
                tier_id=record.tier_id,
                state_id=record.state_id,
                front_path=front_dst,
                back_path=back_dst,
                source_url=record.source_url,
            )
        )

    return prepped_records


def prepare_grading_dataset(
    db_path: Path = grading_config.DB_PATH,
    img_base_dir: Path | None = grading_config.IMG_BASE_DIR,
    table: str = grading_config.TABLE_NAME,
    index_out: Path = grading_config.GRADING_INDEX_PATH,
    prepped_img_dir: Path = grading_config.PREPPED_IMG_DIR,
    canvas_size: int = grading_config.IMAGE_SIZE,
    val_frac: float = grading_config.VAL_FRAC,
    seed: int = grading_config.SEED,
) -> tuple[int, int]:
    log.info(f"Loading records from {db_path}")

    records = load_grade_records(
        db_path,
        img_base_dir,
        table=table,
    )

    prepped_records = prep_all_records_offline(
        records,
        prepped_img_dir,
        canvas_size,
    )

    # Held out set
    train_records, val_records = split_by_coin_id(
        prepped_records,
        val_frac=val_frac,
        seed=seed,
    )

    train_designs = len({r.coin_id for r in train_records})
    val_designs = len({r.coin_id for r in val_records})

    log.info(
        f"Split: {len(train_records)} train ({train_designs} designs), "
        f"{len(val_records)} val ({val_designs} designs)"
    )

    save_split(
        train_records,
        val_records,
        index_out,
    )
    log.info(f"Saved split to {index_out}")

    return len(train_records), len(val_records)


if __name__ == "__main__":
    prepare_grading_dataset()