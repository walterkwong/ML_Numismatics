from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import config
import pandas as pd
import torch
from augment import build_eval_transform, build_train_transform
from backbones import build_backbone
from data import CoinDataset, load_index
from model import (
    MultiTaskCoinModel,
    compute_loss,
    evaluate_all,
    measure_latency,
    measure_model_size,
)
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_sweep")


@dataclass
class RunConfig:
    backbone: str
    adapt: str
    output_dir: Path
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr: float = 1e-5
    samples_per_face: int = 8
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def _collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets = {
        "family": torch.tensor([item[1]["family"] for item in batch]),
        "category": torch.tensor([item[1]["category"] for item in batch]),
        "variety": torch.tensor([item[1]["variety"] for item in batch]),
        "face_value": torch.tensor([item[1]["face_value"] for item in batch]),
        "year": torch.tensor(
            [item[1]["year"] for item in batch],
            dtype=torch.float32,
        ),
        "seen_in_training": [
            item[1]["seen_in_training"] for item in batch
        ],
    }

    return images, targets


def _save_checkpoint(
    model: MultiTaskCoinModel,
    adapt_mode: str,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    heads_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    torch.save(heads_state, path / "heads.pt")

    if adapt_mode == "lora":
        model.backbone.save_pretrained(path / "lora_adapter")
    elif adapt_mode == "full":
        torch.save(model.backbone.state_dict(), path / "backbone.pt")


def run_training_experiment(cfg: RunConfig) -> dict:
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Running {cfg.backbone} ({cfg.adapt})")

    backbone, embed_dim, image_size, mean, std = build_backbone(
        cfg.backbone,
        cfg.adapt,
    )

    records, label_maps = load_index(config.DATASET_INDEX_PATH)

    train_records = [r for r in records if r.seen_in_training]
    heldout_records = [r for r in records if not r.seen_in_training]

    train_tf = build_train_transform(image_size, mean, std)
    eval_tf = build_eval_transform(image_size, mean, std)

    train_ds = CoinDataset(
        train_records,
        train_tf,
        samples_per_face=cfg.samples_per_face,
    )
    val_gallery_ds = CoinDataset(
        heldout_records,
        eval_tf,
        samples_per_face=1,
    )
    val_query_ds = CoinDataset(
        heldout_records,
        train_tf,
        samples_per_face=4,
    )

    val_gallery_records = [item[0] for item in val_gallery_ds.items]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )

    model = MultiTaskCoinModel(
        backbone=backbone,
        embed_dim=embed_dim,
        num_families=label_maps.num_families,
        num_categories=label_maps.num_categories,
        num_varieties=label_maps.num_varieties,
        num_face_values=label_maps.num_face_values,
    ).to(device)

    head_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("backbone.") and param.requires_grad
    ]
    backbone_params = [
        param
        for param in model.backbone.parameters()
        if param.requires_grad
    ]

    param_groups = [{"params": head_params, "lr": cfg.lr}]

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": cfg.backbone_lr,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    best_bayes_rank1 = -1.0
    history = []

    for epoch in range(cfg.epochs):
        model.train()

        if cfg.adapt == "linear":
            model.backbone.eval()

        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in targets.items()
            }

            outputs = model(images)
            loss, _ = compute_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        metrics = evaluate_all(
            model=model,
            gallery_ds=val_gallery_ds,
            query_ds=val_query_ds,
            gallery_records=val_gallery_records,
            num_categories=label_maps.num_categories,
            num_face_values=label_maps.num_face_values,
            device=device,
            num_workers=cfg.num_workers,
        )

        log.info(
            f"epoch {epoch + 1:2d}/{cfg.epochs} | loss {avg_loss:.3f} | "
            f"Rank-1: Base={metrics['baseline_retrieval_rank1']:.3f} -> "
            f"Bayes={metrics['hierarchical_bayes_rank1']:.3f} | "
            f"Cat: Base={metrics['category_raw_top1']:.3f} -> "
            f"Bayes={metrics['category_bayes_top1']:.3f} | "
            f"YearMAE: Base={metrics['year_raw_mae_years']:.1f}y -> "
            f"Bayes={metrics['year_bayes_mae_years']:.1f}y | "
            f"{time.time() - epoch_start:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            **metrics,
        })

        if metrics["hierarchical_bayes_rank1"] > best_bayes_rank1:
            best_bayes_rank1 = metrics["hierarchical_bayes_rank1"]
            _save_checkpoint(
                model,
                cfg.adapt,
                cfg.output_dir / "best",
            )

    _save_checkpoint(
        model,
        cfg.adapt,
        cfg.output_dir / "final",
    )

    latency_res = measure_latency(
        model,
        device,
        image_size,
    )
    size_res = measure_model_size(model)

    result = {
        "backbone": cfg.backbone,
        "adapt": cfg.adapt,
        "best_bayes_rank1": best_bayes_rank1,
        **history[-1],
        **latency_res,
        **size_res,
    }

    (cfg.output_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def run_all_experiments() -> None:
    experiments = [
        ("dinov2", "linear"),
        ("dinov2", "lora"),
        ("ijepa", "linear"),
        ("ijepa", "lora"),
        ("clip", "linear"),
        ("clip", "lora"),
        ("convnext_tiny", "full"),
    ]

    runs_dir = config.PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for backbone, adapt in experiments:
        cfg = RunConfig(
            backbone=backbone,
            adapt=adapt,
            output_dir=runs_dir / f"{backbone}_{adapt}",
            epochs=15,
            batch_size=32,
            samples_per_face=8,
        )

        try:
            result = run_training_experiment(cfg)
            all_results.append(result)
        except Exception as exc:
            log.exception(
                f"Experiment {backbone}/{adapt} failed: {exc}",
            )
            all_results.append({
                "backbone": backbone,
                "adapt": adapt,
                "error": str(exc),
            })

    df = pd.DataFrame(all_results)
    df.to_csv(
        runs_dir / "benchmark_comparison.csv",
        index=False,
    )

    report_cols = [
        "backbone",
        "adapt",
        "baseline_retrieval_rank1",
        "hierarchical_bayes_rank1",
        "category_raw_top1",
        "category_bayes_top1",
        "year_raw_mae_years",
        "year_bayes_mae_years",
        "latency_mean_ms",
    ]
    display_cols = [col for col in report_cols if col in df.columns]

    print("\nBenchmark comparison:")
    print(
        df[display_cols]
        .sort_values("hierarchical_bayes_rank1", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_all_experiments()