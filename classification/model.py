from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from backbones import pooled_forward
from data import CoinRecord, denormalize_year
from torch import nn
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskCoinModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_families: int,
        num_categories: int,
        num_varieties: int,
        num_face_values: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.family_head = MLPHead(embed_dim, hidden_dim, num_families)
        self.category_head = MLPHead(embed_dim, hidden_dim, num_categories)
        self.variety_head = MLPHead(embed_dim, hidden_dim, num_varieties)
        self.face_value_head = MLPHead(embed_dim, hidden_dim, num_face_values)
        self.year_head = MLPHead(embed_dim, hidden_dim, 1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = pooled_forward(self.backbone, pixel_values)

        return {
            "family": self.family_head(emb),
            "category": self.category_head(emb),
            "variety": self.variety_head(emb),
            "face_value": self.face_value_head(emb),
            "year": self.year_head(emb).squeeze(-1),
            "embedding": emb,
        }


DEFAULT_TASK_WEIGHTS = {
    "family": 1.0,
    "category": 1.0,
    "variety": 2.0,
    "face_value": 0.5,
    "year": 0.5,
}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    criterion_focal = FocalLoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Focal for imbalanced classes
    losses = {
        "family": criterion_focal(outputs["family"], targets["family"]),
        "category": criterion_focal(outputs["category"], targets["category"]),
        "variety": criterion_ce(outputs["variety"], targets["variety"]),
        "face_value": outputs["face_value"].sum() * 0.0,
        "year": outputs["year"].sum() * 0.0,
    }

    # Ignore missing targets
    fv_mask = targets["face_value"] >= 0
    if fv_mask.any():
        losses["face_value"] = criterion_focal(
            outputs["face_value"][fv_mask],
            targets["face_value"][fv_mask],
        )

    year_mask = ~torch.isnan(targets["year"])
    if year_mask.any():
        losses["year"] = F.mse_loss(
            outputs["year"][year_mask],
            targets["year"][year_mask].float(),
        )

    total = sum(
        DEFAULT_TASK_WEIGHTS[key] * losses[key]
        for key in losses
    )

    return total, {
        key: value.detach().item()
        for key, value in losses.items()
    }


def year_mae_years(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
) -> float:
    mask = ~torch.isnan(target_norm)

    if not mask.any():
        return float("nan")

    pred_years = torch.tensor([
        denormalize_year(v.item())
        for v in pred_norm[mask]
    ])
    true_years = torch.tensor([
        denormalize_year(v.item())
        for v in target_norm[mask]
    ])

    return (pred_years - true_years).abs().mean().item()


def compute_hierarchical_posteriors(
    sim_matrix: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    gallery_records: list[CoinRecord],
    num_categories: int,
    num_face_values: int,
    year_sigma: float = 8.0,
) -> dict[str, torch.Tensor]:
    p_family = F.softmax(outputs["family"], dim=-1).cpu()
    p_category = F.softmax(outputs["category"], dim=-1).cpu()
    p_face_value = F.softmax(outputs["face_value"], dim=-1).cpu()

    pred_years = torch.tensor([
        denormalize_year(y.item())
        for y in outputs["year"].cpu()
    ])

    # Convert similarity to probabilities
    p_visual = F.softmax(sim_matrix / 0.07, dim=-1).cpu()

    category_to_family = torch.zeros(
        num_categories,
        dtype=torch.long,
    )
    for record in gallery_records:
        category_to_family[record.category_id] = record.family_id

    p_family_for_category = p_family[:, category_to_family]

    # Category posterior
    p_category_bayes = F.softmax(
        torch.log(p_category + 1e-8)
        + torch.log(p_family_for_category + 1e-8),
        dim=-1,
    )

    category_face_value = torch.zeros(
        (num_categories, num_face_values)
    )
    for record in gallery_records:
        if record.face_value_id is not None and record.face_value_id >= 0:
            category_face_value[
                record.category_id,
                record.face_value_id,
            ] = 1.0

    # Face value posterior
    face_value_prior = p_category_bayes @ category_face_value
    p_face_value_bayes = F.softmax(
        torch.log(p_face_value + 1e-8)
        + torch.log(face_value_prior + 1e-8),
        dim=-1,
    )

    family_ids = torch.tensor([
        r.family_id for r in gallery_records
    ])
    category_ids = torch.tensor([
        r.category_id for r in gallery_records
    ])
    face_value_ids = torch.tensor([
        r.face_value_id if r.face_value_id is not None else -1
        for r in gallery_records
    ])
    gallery_years = torch.tensor([
        r.year if r.year is not None else float("nan")
        for r in gallery_records
    ])

    prob_family = p_family[:, family_ids]
    prob_category = p_category[:, category_ids]

    # Ignore missing face values
    prob_face_value = torch.ones_like(p_visual)
    valid_face_value = face_value_ids >= 0
    prob_face_value[:, valid_face_value] = p_face_value[
        :, face_value_ids[valid_face_value]
    ]

    # Ignore missing years
    prob_year = torch.ones_like(p_visual)
    valid_year = ~torch.isnan(gallery_years)
    valid_query_year = ~torch.isnan(pred_years)

    diff = (
        pred_years.unsqueeze(1)
        - gallery_years[valid_year].unsqueeze(0)
    )

    year_prob = torch.exp(-0.5 * (diff / year_sigma) ** 2)
    year_prob = torch.where(
        valid_query_year.unsqueeze(1),
        year_prob,
        torch.ones_like(year_prob),
    )

    prob_year[:, valid_year] = torch.clamp(year_prob, min=1e-4)

    p_variety_bayes = F.softmax(
        torch.log(p_visual + 1e-8)
        + torch.log(prob_family + 1e-8)
        + torch.log(prob_category + 1e-8)
        + torch.log(prob_face_value + 1e-8)
        + torch.log(prob_year + 1e-8),
        dim=-1,
    )

    # Expected year from posterior
    safe_years = gallery_years.clone()
    safe_years[torch.isnan(safe_years)] = 0.0

    bayes_expected_year = (
        p_variety_bayes * safe_years.unsqueeze(0)
    ).sum(dim=-1)

    return {
        "p_category_bayes": p_category_bayes,
        "p_fv_bayes": p_face_value_bayes,
        "p_variety_bayes": p_variety_bayes,
        "bayes_expected_year": bayes_expected_year,
    }


@torch.no_grad()
def evaluate_all(
    model: nn.Module,
    gallery_ds,
    query_ds,
    gallery_records: list[CoinRecord],
    num_categories: int,
    num_face_values: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, float]:
    model.eval()

    gallery_loader = DataLoader(
        gallery_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    gallery_embs = []
    gallery_labels = []

    for images, targets in gallery_loader:
        embeddings = model(images.to(device))["embedding"]
        gallery_embs.append(F.normalize(embeddings, dim=-1).cpu())
        gallery_labels.append(targets["variety"])

    gallery_embs = torch.cat(gallery_embs)
    gallery_labels = torch.cat(gallery_labels)

    query_loader = DataLoader(
        query_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    query_embs = []
    query_labels = []

    output_keys = ["family", "category", "face_value", "year"]
    query_outputs = {key: [] for key in output_keys}
    query_targets = {key: [] for key in output_keys}

    for images, targets in query_loader:
        outputs = model(images.to(device))

        query_embs.append(
            F.normalize(outputs["embedding"], dim=-1).cpu()
        )
        query_labels.append(targets["variety"])

        for key in output_keys:
            query_outputs[key].append(outputs[key].cpu())
            query_targets[key].append(targets[key])

    query_embs = torch.cat(query_embs)
    query_labels = torch.cat(query_labels)
    query_outputs = {
        key: torch.cat(value)
        for key, value in query_outputs.items()
    }
    query_targets = {
        key: torch.cat(value)
        for key, value in query_targets.items()
    }

    sim_matrix = query_embs @ gallery_embs.T

    top5 = sim_matrix.topk(
        min(5, gallery_embs.size(0)),
        dim=1,
    ).indices
    base_preds = gallery_labels[top5]

    base_rank1 = (
        (base_preds[:, 0] == query_labels).float().mean().item()
    )
    base_rank5 = (
        (base_preds == query_labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    bayes = compute_hierarchical_posteriors(
        sim_matrix,
        query_outputs,
        gallery_records,
        num_categories,
        num_face_values,
    )

    bayes_top5 = bayes["p_variety_bayes"].topk(
        min(5, gallery_embs.size(0)),
        dim=1,
    ).indices
    bayes_preds = gallery_labels[bayes_top5]

    bayes_rank1 = (
        (bayes_preds[:, 0] == query_labels).float().mean().item()
    )
    bayes_rank5 = (
        (bayes_preds == query_labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    n = query_labels.size(0)

    family_top1 = (
        (query_outputs["family"].argmax(1) == query_targets["family"])
        .sum()
        .item()
        / max(1, n)
    )

    category_raw_top1 = (
        (query_outputs["category"].argmax(1) == query_targets["category"])
        .sum()
        .item()
        / max(1, n)
    )

    category_bayes_top1 = (
        (
            bayes["p_category_bayes"].argmax(1)
            == query_targets["category"]
        )
        .sum()
        .item()
        / max(1, n)
    )

    fv_mask = query_targets["face_value"] >= 0
    fv_raw_top1 = float("nan")
    fv_bayes_top1 = float("nan")

    if fv_mask.any():
        fv_raw_top1 = (
            (
                query_outputs["face_value"][fv_mask].argmax(1)
                == query_targets["face_value"][fv_mask]
            )
            .sum()
            .item()
            / fv_mask.sum().item()
        )

        fv_bayes_top1 = (
            (
                bayes["p_fv_bayes"][fv_mask].argmax(1)
                == query_targets["face_value"][fv_mask]
            )
            .sum()
            .item()
            / fv_mask.sum().item()
        )

    raw_year_mae = year_mae_years(
        query_outputs["year"],
        query_targets["year"],
    )

    target_years = torch.tensor([
        denormalize_year(v.item())
        for v in query_targets["year"]
    ])
    valid_year = ~torch.isnan(query_targets["year"])

    if valid_year.any():
        bayes_year_mae = (
            bayes["bayes_expected_year"][valid_year]
            - target_years[valid_year]
        ).abs().mean().item()
    else:
        bayes_year_mae = float("nan")

    return {
        "baseline_retrieval_rank1": base_rank1,
        "baseline_retrieval_rank5": base_rank5,
        "hierarchical_bayes_rank1": bayes_rank1,
        "hierarchical_bayes_rank5": bayes_rank5,
        "family_top1": family_top1,
        "category_raw_top1": category_raw_top1,
        "category_bayes_top1": category_bayes_top1,
        "face_value_raw_top1": fv_raw_top1,
        "face_value_bayes_top1": fv_bayes_top1,
        "year_raw_mae_years": raw_year_mae,
        "year_bayes_mae_years": bayes_year_mae,
    }


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int,
) -> dict[str, float]:
    model.eval()
    dummy = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    # Warm up
    for _ in range(10):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []

    for _ in range(100):
        start = time.perf_counter()
        model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    times.sort()

    return {
        "latency_mean_ms": sum(times) / len(times),
        "latency_p95_ms": times[int(len(times) * 0.95)],
    }


def measure_model_size(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "size_total_mb_fp32": round(total * 4 / (1024 ** 2), 2),
        "size_trainable_mb_fp32": round(
            trainable * 4 / (1024 ** 2),
            2,
        ),
    }