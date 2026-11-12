import collections
import math

import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score


def session_options():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return so


def make_session(path, disable_graph_opt=False):
    so = session_options()
    if disable_graph_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def optimized_op_counts(path, save_to):
    so = session_options()
    so.optimized_model_filepath = str(save_to)
    ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return dict(collections.Counter(n.op_type for n in onnx.load(str(save_to)).graph.node))


def onnx_logits(sess, x, batch=64):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return np.concatenate([sess.run(["logits"], {"input": x[i : i + batch]})[0] for i in range(0, len(x), batch)])


def metrics(pred, y):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
    }


def paired(pred, ref_pred, y):
    n = len(y)
    ok, ref_ok = pred == y, ref_pred == y
    b = int(np.sum(ok & ~ref_ok))
    c = int(np.sum(~ok & ref_ok))
    within_noise = abs(b - c) <= 2 * math.sqrt(b + c)
    if within_noise:
        verdict = "не відрізняється від шуму"
    else:
        verdict = "краща за FP32" if b > c else "гірша за FP32"
    return {
        "agreement": float(np.mean(pred == ref_pred)),
        "b": b,
        "c": c,
        "delta_acc": (b - c) / n,
        "se_delta": math.sqrt(b + c) / n,
        "threshold_2sqrt": 2 * math.sqrt(b + c),
        "within_noise": within_noise,
        "verdict": verdict,
    }


def row(pred, ref_pred, y):
    return {**metrics(pred, y), **paired(pred, ref_pred, y)}


def single_model_se(p, n):
    return math.sqrt(p * (1 - p) / n)
