# preprocessing: ngc_coins.json -> dataset_index.pkl

import json
import logging

import config
from data import build_label_maps, build_records, save_index, split_records


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def prepare_dataset() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(
        config.SCRAPED_JSON_PATH.read_text(encoding="utf-8")
    )
    log.info(f"Loaded {len(raw_data)} coin entries")

    label_maps = build_label_maps(raw_data)
    log.info(
        f"Labels: {label_maps.num_families} families, "
        f"{label_maps.num_categories} categories, "
        f"{label_maps.num_varieties} varieties, "
        f"{label_maps.num_face_values} face values"
    )

    records = build_records(
        raw_records=raw_data,
        img_dir=config.IMG_CACHE_DIR,
        label_maps=label_maps,
        canvas_size=config.CANVAS_SIZE,
        max_workers=config.MAX_WORKERS,
    )

    train_records, heldout_records = split_records(
        records,
        heldout_frac=config.HELDOUT_FRAC,
        seed=config.SEED,
    )
    log.info(
        f"Split: {len(train_records)} train, "
        f"{len(heldout_records)} held out"
    )

    save_index(
        train_records + heldout_records,
        label_maps,
        config.DATASET_INDEX_PATH,
    )
    log.info(f"Saved dataset index to {config.DATASET_INDEX_PATH}")


if __name__ == "__main__":
    prepare_dataset()