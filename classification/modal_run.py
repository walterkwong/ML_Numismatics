from pathlib import Path

import modal

# https://modal.com/docs

volume = modal.Volume.from_name(
    "coin-dataset-vol",
    create_if_missing=True,
)
VOLUME_MOUNT_PATH = "/vol/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "transformers",
        "peft",
        "albumentations",
        "opencv-python-headless",
        "pandas",
        "requests",
    )
    .add_local_python_source(
        "config",
        "data",
        "augment",
        "backbones",
        "model",
        "train",
        "image_prep",
        "prepare_dataset",
    )
    .add_local_file("ngc_coins.json", "/root/ngc_coins.json")
)

app = modal.App("coin-classification-suite", image=image)


@app.function(
    volumes={VOLUME_MOUNT_PATH: volume},
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def prepare_data_remote():
    import config
    from prepare_dataset import prepare_dataset

    config.DATA_DIR = Path(VOLUME_MOUNT_PATH)
    config.SCRAPED_JSON_PATH = Path("/root/ngc_coins.json")
    config.IMG_CACHE_DIR = config.DATA_DIR / "images"
    config.DATASET_INDEX_PATH = config.DATA_DIR / "dataset_index.pkl"

    print("Starting dataset preparation...")
    prepare_dataset()

    volume.commit()
    print("Dataset preparation complete.")


@app.function(
    gpu="A100",
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=86400,
    cpu=8.0,
    memory=32768,
)
def train_sweep_remote():
    import config
    from train import run_all_experiments

    config.DATA_DIR = Path(VOLUME_MOUNT_PATH)
    config.DATASET_INDEX_PATH = config.DATA_DIR / "dataset_index.pkl"
    config.PROJECT_ROOT = config.DATA_DIR

    print("Starting sweep...")
    volume.reload()

    run_all_experiments()

    volume.commit()
    print("Sweep complete.")


@app.local_entrypoint()
def main(action: str = "train", detached: bool = True):
    if action == "prepare":
        print("Starting data preparation on Modal...")
        prepare_data_remote.remote()

    elif action == "train":
        print("Running training sweep...")
        train_sweep_remote.remote()