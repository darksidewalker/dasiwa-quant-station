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


def _factor_relative_error(source, first, second, kind):
    error = 0.; norm = 0.
    for start in range(0, source.shape[0], 256):
        end = min(source.shape[0], start + 256)
        if kind == 'lora':
            rebuilt = second[start:end].float() @ first.float()
        else:
            indices = torch.arange(start, end, device=source.device)
            rebuilt = (first[indices // second.shape[0], :, None].float() *
                       second[indices % second.shape[0], None, :].float()).reshape(end-start, -1)
        rows = source[start:end].double()
        error += float((rows - rebuilt.double()).square().sum())
        norm += float(rows.square().sum())
    return (error / max(norm, 1e-24)) ** 0.5


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
    retained = 1.0 if total <= 0 else float((singular[:rank].square().sum() / total).item())
    return down, up, FactorizationReport("lora", rank, retained, _factor_relative_error(work, down, up, 'lora'))


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
    total = singular.square().sum()
    retained = 1.0 if total <= 0 else float((singular[0].square() / total).item())
    return w1, w2, FactorizationReport("lokr", 1, retained, _factor_relative_error(work, w1, w2, 'lokr'))
def factorize_additive_lora(pairs, *, max_rank=0, energy=0.99):
    """Exact QR/core SVD of sum(scale * B @ A), without a dense delta."""
    down = torch.cat([a.to(torch.float32) for a, b, scale in pairs], dim=0)
    up = torch.cat([b.to(torch.float32) * scale for a, b, scale in pairs], dim=1)
    if not torch.isfinite(down).all() or not torch.isfinite(up).all():
        raise ValueError('cannot factorize factors containing NaN or infinity')
    qu, ru = torch.linalg.qr(up, mode='reduced')
    qd, rd = torch.linalg.qr(down.T, mode='reduced')
    small_down, small_up, report = factorize_lora(ru @ rd.T, max_rank=max_rank, energy=energy)
    return (small_down @ qd.T).contiguous(), (qu @ small_up).contiguous(), report
