"""Device accounting for bounded H3 row math, never gauge/SVD work."""
import json
import torch


class H3DevicePolicy:
    def __init__(self, payload):
        self.requested = payload.get('merge_device', 'auto')
        raw = str(payload.get('cuda_device', 'cuda:0'))
        self.cuda_device = 'cuda:0' if raw == 'cuda' else raw if raw.startswith('cuda:') else 'cuda:' + raw
        self.headroom = int(payload.get('vram_headroom_mb', 1024)) * 1024**2
        self.counts = dict(cpu=0, cuda=0, cuda_unavailable=0, insufficient_vram=0, oom=0)

    def run(self, operation, tensors, output_elements):
        """Execute one bounded chunk in double precision, returning CPU double."""
        if self.requested != 'cpu':
            try:
                available = torch.cuda.is_available() and int(self.cuda_device.split(':')[1]) < torch.cuda.device_count()
            except Exception:
                available = False
            if not available:
                self.counts['cuda_unavailable'] += 1
            else:
                try:
                    free, _ = torch.cuda.mem_get_info(self.cuda_device)
                except Exception:
                    free = None
                    self.counts['cuda_unavailable'] += 1
                # Double inputs, output and conservative intermediate/workspace reserve.
                needed = 4 * 8 * (sum(t.numel() for t in tensors) + output_elements)
                if free is None:
                    pass
                elif free < self.headroom + needed:
                    self.counts['insufficient_vram'] += 1
                else:
                    try:
                        result = operation(*(t.to(device=self.cuda_device, dtype=torch.float64) for t in tensors)).cpu()
                    except torch.cuda.OutOfMemoryError:
                        self.counts['oom'] += 1
                    else:
                        self.counts['cuda'] += 1
                        return result
                    # Outside except: release traceback-held CUDA inputs before cache cleanup.
                    try:
                        with torch.cuda.device(self.cuda_device):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass  # Cleanup must not prevent the guaranteed CPU retry.
        result = operation(*(t.to(device='cpu', dtype=torch.float64) for t in tensors))
        self.counts['cpu'] += 1
        return result.cpu()

    def summary(self):
        used = 'mixed' if self.counts['cpu'] and self.counts['cuda'] else 'cuda' if self.counts['cuda'] else 'cpu' if self.counts['cpu'] else 'none'
        fallbacks = sum(self.counts[k] for k in ('cuda_unavailable', 'insufficient_vram', 'oom'))
        return dict(requested=self.requested, used=used, cuda_device=self.cuda_device,
                    unit='row_chunks', fallbacks=fallbacks, **self.counts)

    def log(self):
        return 'H3 row device summary: ' + json.dumps(self.summary(), sort_keys=True)
