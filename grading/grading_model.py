from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from backbones import pooled_forward
from grading_data import grade_to_state_id, grade_to_tier_id
from torch import nn


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
        loss = (1.0 - pt) ** self.gamma * ce

        return loss.mean()


class CoralOrdinalHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        grade_levels: list[float],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.grade_levels = sorted(float(g) for g in grade_levels)
        self.num_levels = len(self.grade_levels)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared = nn.Linear(hidden_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_levels - 1))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.trunk(x)) + self.biases

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"grade_logits": self._logits(x)}

    def grade_to_rank(self, grade: torch.Tensor) -> torch.Tensor:
        levels = torch.tensor(
            self.grade_levels,
            device=grade.device,
            dtype=torch.float32,
        )
        return (grade.unsqueeze(1) - levels.unsqueeze(0)).abs().argmin(dim=1)

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        target_grade: torch.Tensor,
    ) -> torch.Tensor:
        logits = out["grade_logits"]
        target_rank = self.grade_to_rank(target_grade)

        thresholds = torch.arange(
            self.num_levels - 1,
            device=logits.device,
        )
        targets = (target_rank.unsqueeze(1) > thresholds).float()

        return F.binary_cross_entropy_with_logits(logits, targets)

    def get_discrete_probabilities(
        self,
        out: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = out["grade_logits"]
        cumulative = torch.sigmoid(logits)

        batch_size = logits.size(0)
        cumulative = torch.cat(
            [
                torch.ones(batch_size, 1, device=logits.device),
                cumulative,
                torch.zeros(batch_size, 1, device=logits.device),
            ],
            dim=1,
        )

        probs = cumulative[:, :-1] - cumulative[:, 1:]
        return probs.clamp_min(1e-8)


class GaussianRegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        grade_levels: list[float],
        max_grade: float = 70.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.grade_levels = sorted(float(g) for g in grade_levels)
        self.max_grade = max_grade

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden_dim, 1)
        self.logvar_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)

        mu = torch.sigmoid(self.mu_layer(h).squeeze(-1)) * self.max_grade
        logvar = self.logvar_layer(h).squeeze(-1).clamp(-4.0, 4.0)

        return {"mu": mu, "logvar": logvar}

    def compute_loss(
        self,
        out: dict[str, torch.Tensor],
        target_grade: torch.Tensor,
    ) -> torch.Tensor:
        mu = out["mu"]
        logvar = out["logvar"]

        precision = torch.exp(-logvar)
        loss = 0.5 * (
            precision * (target_grade - mu) ** 2 + logvar
        )

        return loss.mean()

    def get_discrete_probabilities(
        self,
        out: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mu = out["mu"].unsqueeze(1)
        std = torch.exp(0.5 * out["logvar"]).unsqueeze(1)

        levels = torch.tensor(
            self.grade_levels,
            device=mu.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        probs = torch.exp(
            -0.5 * ((levels - mu) / (std + 1e-6)) ** 2
        )
        probs = probs / (std + 1e-6)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return probs.clamp_min(1e-8)


class MultiTaskGrader(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        head_type: str,
        grade_levels: list[float],
        max_grade: float = 70.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.head_type = head_type
        self.grade_levels = sorted(float(g) for g in grade_levels)

        self.state_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.tier_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10),
        )

        if head_type == "coral":
            self.grade_head = CoralOrdinalHead(
                in_dim,
                hidden_dim,
                self.grade_levels,
                dropout,
            )
        elif head_type == "gaussian":
            self.grade_head = GaussianRegressionHead(
                in_dim,
                hidden_dim,
                self.grade_levels,
                max_grade,
                dropout,
            )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {
            "state_logits": self.state_head(embedding),
            "tier_logits": self.tier_head(embedding),
            "embedding": embedding,
        }
        out.update(self.grade_head(embedding))

        return out


class HandcraftedGrader(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.multi_task = MultiTaskGrader(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.multi_task(self.feature_extractor(x))


class DeepGrader(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.multi_task = MultiTaskGrader(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = pooled_forward(self.backbone, x)
        embedding = self.trunk(embedding)

        return self.multi_task(embedding)


DEFAULT_GRADING_WEIGHTS = {
    "grade": 2.0,
    "tier": 1.0,
    "state": 0.5,
}

_focal_loss = FocalLoss(gamma=2.0)
_ce_loss = nn.CrossEntropyLoss()


def compute_grading_loss(
    model: nn.Module,
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = DEFAULT_GRADING_WEIGHTS

    state_loss = _ce_loss(
        outputs["state_logits"],
        targets["state"],
    )
    tier_loss = _focal_loss(
        outputs["tier_logits"],
        targets["tier"],
    )
    grade_loss = model.multi_task.grade_head.compute_loss(
        outputs,
        targets["grade"],
    )

    total = (
        weights["grade"] * grade_loss
        + weights["tier"] * tier_loss
        + weights["state"] * state_loss
    )

    losses = {
        "total": total.detach().item(),
        "grade": grade_loss.detach().item(),
        "tier": tier_loss.detach().item(),
        "state": state_loss.detach().item(),
    }

    return total, losses


@torch.no_grad()
def compute_bayesian_grade_posteriors(
    model: nn.Module,
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    grade_levels = model.multi_task.grade_levels
    device = outputs["state_logits"].device

    p_state = F.softmax(outputs["state_logits"], dim=-1)
    p_tier = F.softmax(outputs["tier_logits"], dim=-1)
    p_grade = model.multi_task.grade_head.get_discrete_probabilities(outputs)

    tier_mapping = torch.tensor(
        [grade_to_tier_id(g) for g in grade_levels],
        device=device,
        dtype=torch.long,
    )
    state_mapping = torch.tensor(
        [grade_to_state_id(g) for g in grade_levels],
        device=device,
        dtype=torch.long,
    )

    p_tier = p_tier[:, tier_mapping]
    p_state = p_state[:, state_mapping]

    log_posterior = (
        torch.log(p_grade + 1e-8)
        + torch.log(p_tier + 1e-8)
        + torch.log(p_state + 1e-8)
    )
    p_bayes = F.softmax(log_posterior, dim=-1)

    levels = torch.tensor(
        grade_levels,
        device=device,
        dtype=torch.float32,
    )

    raw_indices = p_grade.argmax(dim=-1)
    bayes_indices = p_bayes.argmax(dim=-1)

    return {
        "p_grade_raw": p_grade,
        "p_grade_bayes": p_bayes,
        "pred_raw_mode": levels[raw_indices],
        "pred_raw_expected": (p_grade * levels).sum(dim=-1),
        "pred_bayes_mode": levels[bayes_indices],
        "pred_bayes_expected": (p_bayes * levels).sum(dim=-1),
    }


@torch.no_grad()
def measure_grader_latency(
    model: nn.Module,
    device: torch.device,
    input_dim_or_size: int,
    is_handcrafted: bool = False,
) -> dict[str, float]:
    model.eval()

    if is_handcrafted:
        dummy = torch.randn(
            1,
            input_dim_or_size,
            device=device,
        )
    else:
        dummy = torch.randn(
            1,
            3,
            input_dim_or_size,
            input_dim_or_size,
            device=device,
        )

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


def measure_grader_size(model: nn.Module) -> dict[str, float]:
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