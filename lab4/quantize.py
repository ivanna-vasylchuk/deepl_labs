import math

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class ImageReader(CalibrationDataReader):
    def __init__(self, images):
        self.images = np.ascontiguousarray(images, dtype=np.float32)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.images):
            return None
        x = self.images[self.i : self.i + 1]
        self.i += 1
        return {"input": x}

    def rewind(self):
        self.i = 0


def preprocess(fp32_path, prep_path):
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)


def quantize(prep_path, out_path, calib_images, per_channel, reduce_range=False):
    quantize_static(
        str(prep_path),
        str(out_path),
        ImageReader(calib_images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=CalibrationMethod.MinMax,
    )


def channel_ranges(fp32_path):
    g = onnx.load(str(fp32_path)).graph
    weights = {t.name: t for t in g.initializer}
    convs = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        w = numpy_helper.to_array(weights[n.input[1]])
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        convs.append({"name": n.name, "shape": list(w.shape), "r": r, "log2_r": math.log2(r)})
    median_r = float(np.median([c["r"] for c in convs]))
    return {
        "convs": convs,
        "top3": sorted(convs, key=lambda c: -c["r"])[:3],
        "median_r": median_r,
        "log2_median_r": math.log2(median_r),
    }


def int_ranges(path):
    weights = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(path)).graph.initializer}
    conv_w = [v for v in weights.values() if v.dtype == np.int8 and v.ndim == 4]
    return {
        "max_abs_q_weight": max(int(np.abs(w.astype(np.int32)).max()) for w in conv_w),
        "activation_scales": {k: float(v) for k, v in weights.items() if k.endswith("_scale") and v.ndim == 0},
    }
