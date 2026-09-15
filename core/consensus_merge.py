from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ConsensusSettings:
    name: str
    consensus_type: str
    similarity_threshold: float
    power_alpha: float
    diversity_beta: float
    rescale_norm: bool
    soft_comfort_bandpass: bool


@dataclass
class ConsensusStats:
    groups: int = 0
    singleton_groups: int = 0
    rejected_contributors: int = 0
    equal_weight_fallbacks: int = 0


CONSENSUS_PRESETS: Mapping[str, ConsensusSettings] = {
    "balanced": ConsensusSettings("balanced", "median", 0.0, 2.0, 4.0, True, True),
    "conservative": ConsensusSettings("conservative", "median", 0.35, 1.25, 0.0, False, False),
    "neutral": ConsensusSettings("neutral", "median", 0.0, 2.0, 0.0, True, False),
}


def default_consensus_preset(architecture: str) -> str:
    return "balanced" if architecture == "MiniMax H3" else "conservative"


def resolve_consensus_preset(name: str | None, architecture: str) -> ConsensusSettings:
    resolved = name or default_consensus_preset(architecture)
    try:
        return CONSENSUS_PRESETS[resolved]
    except KeyError as exc:
        choices = ", ".join(CONSENSUS_PRESETS)
        raise ValueError(f"consensus_preset must be one of {choices}; got {resolved!r}") from exc


def merge_consensus_rows(
    contributors: torch.Tensor,
    settings: ConsensusSettings,
    *,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ConsensusStats]:
    if contributors.ndim != 3 or contributors.shape[0] == 0:
        raise ValueError("contributors must have shape [contributors, rows, features]")
    work = contributors.to(torch.float32)
    stats = ConsensusStats(groups=int(work.shape[1]))
    if work.shape[0] == 1:
        stats.singleton_groups = int(work.shape[1])
        result = work[0].clone()
        return (result, stats) if return_stats else result

    consensus = torch.quantile(work, 0.5, dim=0) if settings.consensus_type == "median" else torch.mean(work, dim=0)
    similarities = (
        F.normalize(work, p=2, dim=2, eps=1e-8)
        * F.normalize(consensus, p=2, dim=1, eps=1e-8).unsqueeze(0)
    ).sum(dim=2)
    accepted = similarities >= settings.similarity_threshold
    stats.rejected_contributors = int((~accepted).sum().item())
    safe = similarities.clamp(min=0.0, max=1.0)
    weights = torch.where(accepted, safe.pow(settings.power_alpha), torch.zeros_like(safe))
    if settings.diversity_beta > 0.0:
        distance_base = 1.5 if settings.soft_comfort_bandpass else 1.001
        weights *= (distance_base - safe).clamp(min=0.0).pow(settings.diversity_beta)
    sums = weights.sum(dim=0, keepdim=True)
    fallback = sums <= 0
    stats.equal_weight_fallbacks = int(fallback.sum().item())
    normalized = weights / sums.clamp_min(1e-8)
    normalized = torch.where(
        fallback.expand_as(normalized),
        torch.full_like(normalized, 1.0 / work.shape[0]),
        normalized,
    )
    merged = (work * normalized.unsqueeze(2)).sum(dim=0)
    if settings.rescale_norm:
        target_norm = torch.linalg.vector_norm(work, dim=2).mean(dim=0)
        merged_norm = torch.linalg.vector_norm(merged, dim=1)
        scale = torch.where(
            merged_norm > 0,
            target_norm / merged_norm.clamp_min(1e-8),
            torch.ones_like(merged_norm),
        )
        merged *= scale.unsqueeze(1)
    return (merged, stats) if return_stats else merged
