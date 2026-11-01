"""Paper-kernel MMD port with active gradients and declared degenerate guard.

Kernel construction follows the pinned Agarwal author utility: detach pooled
mean squared distance for bandwidth, five geometric scales, sum kernels, and
use biased within/cross means. Different feature/backbone/tasks remain a port.
"""
import torch


def adaptive_gaussian_mmd(source, target, *, kernel_mul=2., kernel_num=5):
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or
        min(source.shape[0], target.shape[0]) < 1 or not torch.isfinite(source).all() or
        not torch.isfinite(target).all() or source.device != target.device or source.dtype != target.dtype or
        not source.is_floating_point()):
        raise ValueError("Expected compatible finite feature batches")
    if type(kernel_num) is not int or kernel_num < 1 or kernel_mul <= 0:
        raise ValueError("Invalid adaptive kernel scales")
    total = torch.cat((source, target))
    distances = (total[:, None]-total[None, :]).square().sum(-1)
    count = total.shape[0]
    bandwidth = distances.detach().sum()/(count*(count-1))
    # Original utility divides by zero for collapsed features. This explicit
    # stability deviation preserves finite, zero discrepancy for identical data.
    bandwidth = bandwidth.clamp_min(torch.finfo(source.dtype).eps)/kernel_mul**(kernel_num//2)
    kernel = sum(torch.exp(-distances/(bandwidth*kernel_mul**i)) for i in range(kernel_num))
    n = source.shape[0]
    return kernel[:n, :n].mean()+kernel[n:, n:].mean()-2*kernel[:n, n:].mean()
