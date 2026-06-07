"""Phase 0 smoke test 1: environment verification.

Run on an A100/A40 node after `source scripts/activate.sh`.
Expected output (running on A100 80GB):
    torch        2.7.1+cu124
    cuda         12.4
    cuda_ok      True
    device 0     NVIDIA A100 80GB PCIe
    sae-lens     6.39.0
    nnsight      0.6.x
    transformers 4.57.x
"""
from __future__ import annotations
import sys


def main() -> int:
    import torch
    print(f"torch        {torch.__version__}")
    print(f"cuda         {torch.version.cuda}")
    print(f"cuda_ok      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0     {torch.cuda.get_device_name(0)}")
        print(f"vram total   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("!! cuda unavailable — wrong node or torch wheel mismatch")
        return 1

    import sae_lens, nnsight, transformers
    print(f"sae-lens     {sae_lens.__version__}")
    print(f"nnsight      {nnsight.__version__}")
    print(f"transformers {transformers.__version__}")

    # Quick allocation sanity: 1 GB tensor must round-trip cleanly.
    t = torch.zeros(1024 * 1024 * 256, dtype=torch.bfloat16, device="cuda")
    t.fill_(1.0)
    assert t.sum().item() == t.numel(), "GPU tensor sum mismatch"
    del t
    torch.cuda.empty_cache()
    print("gpu rt       OK (1 GB bf16 round-trip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
