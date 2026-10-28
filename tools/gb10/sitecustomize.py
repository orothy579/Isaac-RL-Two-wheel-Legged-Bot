"""GB10 (DGX Spark) interpreter hook, put on PYTHONPATH by ``run_gb10.sh``.

torch 2.7 (CUDA 12.8 nvrtc) cannot compile kernels for sm_121, so the TorchScript GPU
fusers fail with ``nvrtc: error: invalid value for --gpu-architecture`` on the first fused
``@torch.jit.script`` call (co_rl normalizer/exporter, Isaac Lab math utils). This hook
switches the fusers off right after ``torch`` is imported, without importing torch itself
before Isaac Sim starts. TorchScript stays usable (policy export); scripted functions run
unfused.
"""

import importlib.abc
import importlib.util
import sys


def _disable_gpu_fusers(torch) -> None:
    try:
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
            torch._C._jit_set_nvfuser_enabled(False)
    except (AttributeError, RuntimeError):
        pass


class _TorchImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _disable_gpu_fusers(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if "torch" in sys.modules:
    _disable_gpu_fusers(sys.modules["torch"])
else:
    sys.meta_path.insert(0, _TorchImportHook())
