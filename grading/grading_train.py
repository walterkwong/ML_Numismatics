from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import grading_config
import pandas as pd
import torch
from backbones import build_backbone
from grading_augment import (
    build_grading_eval_transform,
    build_grading_train_transform,
)
from grading_data import GradeFeatureDataset, GradeImageDataset, load_split
from grading_model import (
    DeepGrader,
    HandcraftedGrader,
    compute_bayesian_grade_posteriors,
    compute_grading_loss,
    measure_grader_latency,
    measure_grader_size,
)
from handcrafted_features import feature_vector_size
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_grader")


@dataclass
class GraderRunConfig:
    backbone: str
    adapt: str
    head: str
    output_dir: Path
    epochs: int = grading_config.EPOCHS
    batch_size: int = grading_config.BATCH_SIZE
    lr: float = grading_config.LR
    backbone_lr: float = grading_config.BACKBONE_LR
    samples_per_record: int = grading_config.SAMPLES_PER_RECORD
    max_grade: float = grading_config.MAX_GRADE
    n_color_clusters: int = grading_config.N_COLOR_CLUSTERS
    radius_frac: float = grading_config.RADIUS_FRAC
    num_workers: int = grading_config.NUM_WORKERS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = grading_config.SEED


@dataclass
class GraderEvalResult:
    raw_mae: float
    raw_within_1: float
    raw_within_3: float
    raw_within_5: float
    bayes_mae: float
    bayes_within_1: float
    bayes_within_3: float
    bayes_within_5: float
    state_top1: float
    tier_top1: float
    n_samples: int


def _collate(batch):
    inputs = torch.stack([item[0] for item in batch])
    targets = {
        "grade": torch.tensor(
            [item[1]["grade"] for item in batch],
            dtype=torch.float32,
        ),
        "tier": torch.tensor(
            [item[1]["tier"] for item in batch],
            dtype=torch.long,
        ),
        "state": torch.tensor(
            [item[1]["state"] for item in batch],
            dtype=torch.long,
        ),
    }

    return inputs, targets


@torch.no_grad()
def evaluate_grader(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> GraderEvalResult:
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    raw_errors = []
    bayes_errors = []
    correct_states = 0
    correct_tiers = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        target_grades = targets["grade"].to(device)
        target_tiers = targets["tier"].to(device)
        target_states = targets["state"].to(device)

        outputs = model(inputs)
        bayes = compute_bayesian_grade_posteriors(model, outputs)

        raw_errors.append(
            (bayes["pred_raw_expected"] - target_grades).abs().cpu()
        )
        bayes_errors.append(
            (bayes["pred_bayes_expected"] - target_grades).abs().cpu()
        )

        pred_state = outputs["state_logits"].argmax(dim=-1)
        pred_tier = outputs["tier_logits"].argmax(dim=-1)

        correct_states += (pred_state == target_states).sum().item()
        correct_tiers += (pred_tier == target_tiers).sum().item()
        total_samples += target_grades.numel()

    if total_samples == 0:
        return GraderEvalResult(
            raw_mae=float("nan"),
            raw_within_1=0.0,
            raw_within_3=0.0,
            raw_within_5=0.0,
            bayes_mae=float("nan"),
            bayes_within_1=0.0,
            bayes_within_3=0.0,
            bayes_within_5=0.0,
            state_top1=0.0,
            tier_top1=0.0,
            n_samples=0,
        )

    raw_errors = torch.cat(raw_errors)
    bayes_errors = torch.cat(bayes_errors)

    return GraderEvalResult(
        raw_mae=raw_errors.mean().item(),
        raw_within_1=(raw_errors <= 1.0).float().mean().item(),
        raw_within_3=(raw_errors <= 3.0).float().mean().item(),
        raw_within_5=(raw_errors <= 5.0).float().mean().item(),
        bayes_mae=bayes_errors.mean().item(),
        bayes_within_1=(bayes_errors <= 1.0).float().mean().item(),
        bayes_within_3=(bayes_errors <= 3.0).float().mean().item(),
        bayes_within_5=(bayes_errors <= 5.0).float().mean().item(),
        state_top1=correct_states / total_samples,
        tier_top1=correct_tiers / total_samples,
        n_samples=total_samples,
    )


def _save_grader_checkpoint(
    model: torch.nn.Module,
    adapt_mode: str,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)

    heads_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    torch.save(heads_state, path / "heads_and_trunk.pt")

    if not hasattr(model, "backbone"):
        return

    if adapt_mode == "lora":
        model.backbone.save_pretrained(path / "lora_adapter")
    elif adapt_mode == "full":
        torch.save(model.backbone.state_dict(), path / "backbone.pt")


def run_training(cfg: GraderRunConfig) -> dict:
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Training {cfg.backbone} ({cfg.adapt}) with {cfg.head} head"
    )

    train_records, val_records = load_split(
        grading_config.GRADING_INDEX_PATH
    )
    log.info(
        f"Loaded split: {len(train_records)} train, "
        f"{len(val_records)} validation records"
    )

    grade_levels = sorted({record.grade for record in train_records})

    if cfg.backbone == "handcrafted":
        feature_dim = feature_vector_size(cfg.n_color_clusters)

        train_tf = build_grading_train_transform(
            grading_config.IMAGE_SIZE,
            to_tensor=False,
        )
        eval_tf = build_grading_eval_transform(
            grading_config.IMAGE_SIZE,
            to_tensor=False,
        )

        train_ds = GradeFeatureDataset(
            train_records,
            train_tf,
            samples_per_record=cfg.samples_per_record,
            n_color_clusters=cfg.n_color_clusters,
            radius_frac=cfg.radius_frac,
        )
        val_ds = GradeFeatureDataset(
            val_records,
            eval_tf,
            samples_per_record=1,
            n_color_clusters=cfg.n_color_clusters,
            radius_frac=cfg.radius_frac,
        )

        model = HandcraftedGrader(
            feature_dim=feature_dim,
            hidden_dim=128,
            head_type=cfg.head,
            grade_levels=grade_levels,
            max_grade=cfg.max_grade,
        ).to(device)

        input_size = feature_dim
        is_handcrafted = True

    else:
        backbone, embed_dim, image_size, mean, std = build_backbone(
            cfg.backbone,
            cfg.adapt,
        )

        train_tf = build_grading_train_transform(
            image_size,
            mean=mean,
            std=std,
            to_tensor=True,
        )
        eval_tf = build_grading_eval_transform(
            image_size,
            mean=mean,
            std=std,
            to_tensor=True,
        )

        train_ds = GradeImageDataset(
            train_records,
            train_tf,
            samples_per_record=cfg.samples_per_record,
        )
        val_ds = GradeImageDataset(
            val_records,
            eval_tf,
            samples_per_record=1,
        )

        model = DeepGrader(
            backbone=backbone,
            embed_dim=embed_dim,
            hidden_dim=512,
            head_type=cfg.head,
            grade_levels=grade_levels,
            max_grade=cfg.max_grade,
        ).to(device)

        input_size = image_size
        is_handcrafted = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )

    head_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("backbone.") and param.requires_grad
    ]

    backbone_params = []
    if hasattr(model, "backbone"):
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

    best_bayes_mae = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        model.train()

        if hasattr(model, "backbone") and cfg.adapt == "linear":
            model.backbone.eval()

        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = {
                key: value.to(device)
                for key, value in targets.items()
            }

            outputs = model(inputs)
            loss, _ = compute_grading_loss(
                model,
                outputs,
                targets,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = running_loss / max(1, n_batches)
        metrics = evaluate_grader(
            model,
            val_ds,
            device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        log.info(
            f"epoch {epoch + 1:2d}/{cfg.epochs} | "
            f"loss {avg_loss:.3f} | "
            f"MAE {metrics.raw_mae:.2f} -> {metrics.bayes_mae:.2f} | "
            f"within-3 {metrics.raw_within_3:.3f} -> "
            f"{metrics.bayes_within_3:.3f} | "
            f"state {metrics.state_top1:.3f} | "
            f"tier {metrics.tier_top1:.3f} | "
            f"{time.time() - epoch_start:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            **asdict(metrics),
        })

        if metrics.bayes_mae < best_bayes_mae:
            best_bayes_mae = metrics.bayes_mae
            _save_grader_checkpoint(
                model,
                cfg.adapt,
                cfg.output_dir / "best",
            )

    _save_grader_checkpoint(
        model,
        cfg.adapt,
        cfg.output_dir / "final",
    )

    latency = measure_grader_latency(
        model,
        device,
        input_size,
        is_handcrafted=is_handcrafted,
    )
    size = measure_grader_size(model)

    result = {
        "backbone": cfg.backbone,
        "adapt": cfg.adapt if cfg.backbone != "handcrafted" else "n/a",
        "head": cfg.head,
        "best_bayes_mae": best_bayes_mae,
        **history[-1],
        **latency,
        **size,
    }

    (cfg.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )

    return result


def run_all_grader_experiments() -> None:
    experiments: list[tuple[str, str, str]] = [
        ("handcrafted", "n/a", "coral"),
        ("handcrafted", "n/a", "gaussian"),
        ("dinov2", "linear", "coral"),
        ("dinov2", "linear", "gaussian"),
        ("dinov2", "lora", "coral"),
        ("dinov2", "lora", "gaussian"),
        ("ijepa", "linear", "coral"),
        ("ijepa", "linear", "gaussian"),
        ("ijepa", "lora", "coral"),
        ("ijepa", "lora", "gaussian"),
        ("convnext_tiny", "full", "coral"),
        ("convnext_tiny", "full", "gaussian"),
        ("clip", "linear", "coral"),
        ("clip", "lora", "coral"),
    ]

    runs_dir = grading_config.PROJECT_ROOT / "runs_grader"
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for backbone, adapt, head in experiments:
        cfg = GraderRunConfig(
            backbone=backbone,
            adapt=adapt,
            head=head,
            output_dir=runs_dir / f"{backbone}_{adapt}_{head}",
            epochs=20,
        )

        try:
            result = run_training(cfg)
            all_results.append(result)
        except Exception as exc:
            log.exception(
                f"Experiment {backbone}/{adapt}/{head} failed: {exc}",
            )
            all_results.append({
                "backbone": backbone,
                "adapt": adapt,
                "head": head,
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
        "head",
        "raw_mae",
        "bayes_mae",
        "raw_within_1",
        "bayes_within_1",
        "raw_within_3",
        "bayes_within_3",
        "raw_within_5",
        "bayes_within_5",
        "state_top1",
        "tier_top1",
        "latency_mean_ms",
    ]
    display_cols = [col for col in report_cols if col in df.columns]

    print("\nBenchmark comparison:")
    print(
        df[display_cols]
        .sort_values("bayes_mae", ascending=True)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_all_grader_experiments()