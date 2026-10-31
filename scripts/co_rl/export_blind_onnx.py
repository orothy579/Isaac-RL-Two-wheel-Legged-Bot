"""Export the BLIND stair actor (co_rl ``model_*.pt`` or a cosim actor ``.npz``) to ONNX and check it.

Input ``obs`` (batch, 804): 50 stacked proprioceptive frames, newest first, then the command (2 vx, 0, 0, 0); each
frame = shoulder positions (2), 0.15 x shoulder and wheel velocities (4), 0.25 x base angular velocity (3), projected
gravity (3), last action (4). Output ``actions`` (batch, 4): the deterministic mean, [left, right shoulder target rad,
left, right wheel speed / 40 rad/s]; clip to [-1, 1] before use, as in training.

    <venv python> scripts/co_rl/export_blind_onnx.py <model.pt | actor.npz> <out.onnx>
"""

import json
import sys

import numpy as np
import torch
from torch import nn


def load_actor(path: str) -> nn.Sequential:
    if path.endswith(".npz"):
        z = np.load(path)
        ws = [(torch.as_tensor(z[f"W{i}"], dtype=torch.float32), torch.as_tensor(z[f"b{i}"], dtype=torch.float32))
              for i in range(4)]
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        ws = [(sd[f"actor.{j}.weight"].float(), sd[f"actor.{j}.bias"].float()) for j in (0, 2, 4, 6)]
    layers = []
    for i, (W, b) in enumerate(ws):
        lin = nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data.copy_(W)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(ws) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def main():
    src, out = sys.argv[1], sys.argv[2]
    actor = load_actor(src)
    n_in = actor[0].in_features
    dummy = torch.zeros(1, n_in)
    torch.onnx.export(actor, dummy, out, input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    report = {"source": src, "onnx": out, "inputs": n_in, "outputs": actor[-1].out_features}
    x = torch.randn(256, n_in)
    with torch.no_grad():
        ref = actor(x).numpy()
    try:
        import onnxruntime as ort

        got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"obs": x.numpy()})[0]
        report["max_abs_diff_vs_torch"] = float(np.abs(got - ref).max())
    except ImportError:
        report["max_abs_diff_vs_torch"] = "onnxruntime not installed"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
