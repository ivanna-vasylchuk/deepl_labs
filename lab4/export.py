import numpy as np
import onnx
import torch
import torchvision

from .data import CLASSES

DEVICE = torch.device("cpu")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if list(ck["classes"]) != CLASSES:
        raise SystemExit(f"Порядок класів у моделі {ck['classes']} не збігається з {CLASSES}")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"])
    model.to(DEVICE, dtype=torch.float32).eval()
    return model, ck


def export_onnx(model, img_size, out_path):
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32, device=DEVICE)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def torch_logits(model, x, batch=64):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch], dtype=np.float32)).to(DEVICE)
            outs.append(model(xb).numpy())
    return np.concatenate(outs)


def count_ops(path):
    ops = [n.op_type for n in onnx.load(str(path)).graph.node]
    return {"Conv": ops.count("Conv"), "BatchNormalization": ops.count("BatchNormalization")}


def arithmetic_intensity(path, bytes_per_value):
    # I = операції / (байти ваг + входу + виходу) при B = 1, як у моделі даху з лекції 9
    g = onnx.shape_inference.infer_shapes(onnx.load(str(path))).graph
    shapes = {
        vi.name: [d.dim_value or 1 for d in vi.type.tensor_type.shape.dim]
        for vi in list(g.value_info) + list(g.input) + list(g.output)
    }
    weights = {t.name: list(t.dims) for t in g.initializer}
    out = []
    for n in g.node:
        if n.op_type not in ("Conv", "Gemm"):
            continue
        w, x, y = weights[n.input[1]], shapes[n.input[0]], shapes[n.output[0]]
        macs = int(np.prod(w)) * (y[2] * y[3] if n.op_type == "Conv" else 1)
        nbytes = bytes_per_value * int(np.prod(w) + np.prod(x) + np.prod(y))
        out.append({"name": n.name, "ops": 2 * macs, "bytes": nbytes, "intensity": 2 * macs / nbytes})
    return sorted(out, key=lambda d: d["intensity"])
