from dataclasses import dataclass

import torch
from scipy import linalg


@dataclass(frozen=True)
class FactorizationReport:
    kind: str
    rank: int
    retained_energy: float
    relative_error: float


def _relative_error(source: torch.Tensor, rebuilt: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(source).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(source - rebuilt) / denominator).item())


def _normalize_pair_sign(first: torch.Tensor, second: torch.Tensor) -> None:
    flat = first.reshape(-1)
    if flat.numel() and flat[torch.argmax(torch.abs(flat))] < 0:
        first.neg_()
        second.neg_()


def reconstruct_lora(down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return up.to(torch.float32) @ down.to(torch.float32)


def factorize_lora(
    delta: torch.Tensor,
    *,
    max_rank: int = 0,
    energy: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, FactorizationReport]:
    if delta.ndim != 2:
        raise ValueError("standard LoRA output requires a 2D delta")
    if not 0.0 < energy <= 1.0 or max_rank < 0:
        raise ValueError("energy must be in (0, 1] and max_rank must be non-negative")
    work = delta.to(torch.float32)
    if not torch.isfinite(work).all():
        raise ValueError("cannot factorize a LoRA delta containing NaN or infinity")
    try:
        u, singular, vh = torch.linalg.svd(work, full_matrices=False)
    except RuntimeError as exc:
        if "linalg.svd" not in str(exc):
            raise
        # PyTorch's CPU SVD uses divide-and-conquer (gesdd), which can fail
        # to converge on ill-conditioned adapter deltas. QR-based gesvd is
        # slower but is a more robust fallback for this case.
        u_np, singular_np, vh_np = linalg.svd(
            work.cpu().numpy(), full_matrices=False, lapack_driver="gesvd", check_finite=False
        )
        u = torch.from_numpy(u_np).to(work.device)
        singular = torch.from_numpy(singular_np).to(work.device)
        vh = torch.from_numpy(vh_np).to(work.device)
    total = singular.square().sum()
    if total <= 0:
        rank = 1
    else:
        cumulative = torch.cumsum(singular.square(), dim=0) / total
        rank = int(torch.searchsorted(cumulative, torch.tensor(energy, device=cumulative.device)).item()) + 1
    if max_rank:
        rank = min(rank, max_rank)
    root = torch.sqrt(singular[:rank])
    up = (u[:, :rank] * root.unsqueeze(0)).contiguous()
    down = (root.unsqueeze(1) * vh[:rank, :]).contiguous()
    for index in range(rank):
        _normalize_pair_sign(up[:, index], down[index, :])
    rebuilt = reconstruct_lora(down, up)
    retained = 1.0 if total <= 0 else float((singular[:rank].square().sum() / total).item())
    return down, up, FactorizationReport("lora", rank, retained, _relative_error(work, rebuilt))


def reconstruct_lokr(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return torch.kron(w1.to(torch.float32), w2.to(torch.float32))


def factorize_lokr(
    delta: torch.Tensor,
    w1_shape: tuple[int, int],
    w2_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, FactorizationReport]:
    if delta.ndim != 2 or len(w1_shape) != 2 or len(w2_shape) != 2:
        raise ValueError("direct LoKr output requires 2D delta and anchor shapes")
    expected = (w1_shape[0] * w2_shape[0], w1_shape[1] * w2_shape[1])
    if tuple(delta.shape) != expected:
        raise ValueError(f"anchor shapes {w1_shape} and {w2_shape} require delta shape {expected}")
    work = delta.to(torch.float32)
    rearranged = work.reshape(w1_shape[0], w2_shape[0], w1_shape[1], w2_shape[1])
    rearranged = rearranged.permute(0, 2, 1, 3).reshape(w1_shape[0] * w1_shape[1], -1)
    u, singular, vh = torch.linalg.svd(rearranged, full_matrices=False)
    root = torch.sqrt(singular[0])
    w1 = (u[:, 0] * root).reshape(w1_shape).contiguous()
    w2 = (vh[0, :] * root).reshape(w2_shape).contiguous()
    _normalize_pair_sign(w1, w2)
    rebuilt = reconstruct_lokr(w1, w2)
    total = singular.square().sum()
    retained = 1.0 if total <= 0 else float((singular[0].square() / total).item())
    return w1, w2, FactorizationReport("lokr", 1, retained, _relative_error(work, rebuilt))
