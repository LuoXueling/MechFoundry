import os as _os
import warp as wp

wp.init()
wp.config.log_level = wp.LOG_ERROR
wp.config.use_precompiled_headers = (
    True  # NVRTC PCH: saves 50-500ms per module (CUDA 12.8+)
)
wp.config.optimization_level = 2  # ~30% faster compile vs default level 3 (CUDA 12.9+)

from . import (
    utils,
    nitsche_kernels,
    nitsche_field,
    condensation,
    preconditioners,
    analytical_sdf,
)  # noqa: F401

_n_workers = min(max(_os.cpu_count() // 2, 1), 16)
wp.load_module(recursive=True, max_workers=_n_workers)

WARP_AVAILABLE = True
