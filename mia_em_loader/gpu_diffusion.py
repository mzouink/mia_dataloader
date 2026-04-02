"""GPU-accelerated flow computation — thin wrapper around topo.

All core logic now lives in ``topo.flow_gpu``.  This module re-exports
the public API so that existing imports continue to work.
"""

from topo.flow_gpu import (
    compute_flow_targets_gpu,
    generate_direct_flows_gpu,
    generate_diffusion_flows_gpu,
)

__all__ = [
    "compute_flow_targets_gpu",
    "generate_direct_flows_gpu",
    "generate_diffusion_flows_gpu",
]
