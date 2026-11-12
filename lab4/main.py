import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data, envinfo, export, latency, quality, quantize

ALL_STEPS = ["1", "2", "3", "4r", "4", "5", "diag"]


def parse_args():
    p = argparse.ArgumentParser(description="ЛР4: FP32 проти INT8")
    p.add_argument("--model", required=True, help="шлях до lab3_model.pt")
    p.add_argument("--split", required=True, help="шлях до split.csv")
    p.add_argument("--data", required=True, help="тека набору з seg_train/ і seg_test/")
    p.add_argument("--out", required=True, help="тека для .onnx і lab4_results.json")
    p.add_argument("--steps", default=",".join(ALL_STEPS), help="пункти через кому, за замовчуванням усі")
    return p.parse_args()


def file_mb(path):
    return Path(path).stat().st_size / 1e6


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def main():
    args = parse_args()
    steps = args.steps.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"lab4_{k}.onnx" for k in ["fp32", "prep", "int8", "int8_per_tensor", "int8_reduce_range"]}
    res_path = out / "lab4_results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}

    def save():
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    res["command_args"] = vars(args)
    res["environment"] = envinfo.collect()

    rows = data.read_split(args.split)
    res["split_control"] = data.check_split(rows)
    print("Контроль поділу: OK")

    model, ck = export.load_model(args.model)
    img_size = int(ck["img_size"])
    tf = data.make_transform(img_size)
    res["model"] = {"regime": ck["regime"], "seed": ck["seed"], "classes": list(ck["classes"]), "img_size": img_size}

    test_rows = data.part_rows(rows, "test")
    y = data.labels(test_rows)
    x_test = data.load_images(args.data, test_rows, tf) if {"1", "3", "4", "diag"} & set(steps) else None

    calib_cache = {}

    def calib_images():
        if "x" not in calib_cache:
            calib = data.calibration_rows(rows)
            calib_cache["check"] = data.check_calibration(calib)
            calib_cache["files"] = [r["file"] for r in calib]
            calib_cache["x"] = data.load_images(args.data, calib, tf)
        return calib_cache["x"]

    fp32_cache = {}

    def fp32_pred():
        if "pred" not in fp32_cache:
            fp32_cache["pred"] = quality.onnx_logits(quality.make_session(paths["fp32"]), x_test).argmax(1)
        return fp32_cache["pred"]

    def ensure_prep():
        if not paths["prep"].exists():
            quantize.preprocess(paths["fp32"], paths["prep"])

    if "1" in steps:
        export_s = timed(export.export_onnx, model, img_size, paths["fp32"])
        z_pt = export.torch_logits(model, x_test)
        z_onnx = quality.onnx_logits(quality.make_session(paths["fp32"]), x_test)
        p_pt, p_onnx = z_pt.argmax(1), z_onnx.argmax(1)
        fp32_cache["pred"] = p_onnx
        m_pt, m_onnx = quality.metrics(p_pt, y), quality.metrics(p_onnx, y)
        res["step1_export"] = {
            "export_time_s": export_s,
            "max_abs_logit_diff": float(np.abs(z_onnx - z_pt).max()),
            "argmax_agreement": float(np.mean(p_pt == p_onnx)),
            "macro_f1_pytorch": m_pt["macro_f1"],
            "macro_f1_onnx_fp32": m_onnx["macro_f1"],
            "accuracy_pytorch": m_pt["accuracy"],
            "node_counts": export.count_ops(paths["fp32"]),
        }
        s = res["step1_export"]
        print(
            f"П.1: max|Δz| = {s['max_abs_logit_diff']:.2e}, збіг argmax {s['argmax_agreement']:.4f}, "
            f"macro-F1 {s['macro_f1_pytorch']:.4f} / {s['macro_f1_onnx_fp32']:.4f}, вузли {s['node_counts']}"
        )
        save()

    if "2" in steps:
        x_cal = calib_images()
        prep_s = timed(quantize.preprocess, paths["fp32"], paths["prep"])
        quant_s = timed(quantize.quantize, paths["prep"], paths["int8"], x_cal, per_channel=True)
        res["step2_calibration"] = {
            **calib_cache["check"],
            "files": calib_cache["files"],
            "preprocess_time_s": prep_s,
            "quantize_static_time_s": quant_s,
            "quantization_total_time_s": prep_s + quant_s,
        }
        print(f"П.2: калібрувальна вибірка OK, pre_process {prep_s:.2f} с, quantize_static {quant_s:.2f} с")
        save()

    if "3" in steps:
        p_fp = fp32_pred()
        p_int8 = quality.onnx_logits(quality.make_session(paths["int8"]), x_test).argmax(1)
        fp_row = quality.row(p_fp, p_fp, y)
        int8_row = quality.row(p_int8, p_fp, y)
        n = len(y)
        single_se = quality.single_model_se(fp_row["accuracy"], n)
        # діагностика поза умовою: без оптимізацій графа QDQ рахується у float, без цілочисельних ядер
        p_no_opt = quality.onnx_logits(quality.make_session(paths["int8"], disable_graph_opt=True), x_test).argmax(1)
        res["step3_quality"] = {
            "n": n,
            "fp32": fp_row,
            "int8": int8_row,
            "single_model_se_fp32": single_se,
            "se_ratio_single_over_paired": single_se / int8_row["se_delta"] if int8_row["se_delta"] > 0 else None,
            "diagnostic_int8_graph_opt_disabled": quality.row(p_no_opt, p_fp, y),
        }
        print(
            f"П.3: INT8 macro-F1 {int8_row['macro_f1']:.4f}, згода {int8_row['agreement']:.4f}, "
            f"b={int8_row['b']} c={int8_row['c']} — {int8_row['verdict']}"
        )
        save()

    if "4r" in steps:
        cr = quantize.channel_ranges(paths["fp32"])
        res["step4_channel_ranges"] = cr
        top = ", ".join(f"{c['name']} {c['r']:.3f}" for c in cr["top3"])
        print(f"П.4: r топ-3: {top}; медіана {cr['median_r']:.3f} ({cr['log2_median_r']:.3f} біта)")
        save()

    if "4" in steps:
        x_cal = calib_images()
        ensure_prep()
        quant_s = timed(quantize.quantize, paths["prep"], paths["int8_per_tensor"], x_cal, per_channel=False)
        p_pt = quality.onnx_logits(quality.make_session(paths["int8_per_tensor"]), x_test).argmax(1)
        res["step4_per_tensor"] = {**quality.row(p_pt, fp32_pred(), y), "quantize_static_time_s": quant_s}
        r = res["step4_per_tensor"]
        print(f"П.4: per-tensor згода {r['agreement']:.4f}, b={r['b']} c={r['c']} — {r['verdict']}")
        save()

    if "5" in steps:
        x1 = data.load_images(args.data, test_rows[:1], tf)
        sessions = {"fp32": quality.make_session(paths["fp32"]), "int8": quality.make_session(paths["int8"])}
        lat = latency.run_protocol(sessions, x1)
        for k in sessions:
            lat[k]["file_mb"] = file_mb(paths[k])
        lat["protocol"] = {
            "rounds": latency.ROUNDS,
            "warmup": latency.WARMUP,
            "measurements": latency.MEASURE,
            "batch": 1,
            "input_image": test_rows[0]["file"],
            "order_in_round": list(sessions),
        }
        lat["speedup_p50"] = lat["fp32"]["p50_ms"] / lat["int8"]["p50_ms"]
        lat["ratio_p95"] = lat["fp32"]["p95_ms"] / lat["int8"]["p95_ms"]
        lat["size_ratio"] = lat["fp32"]["file_mb"] / lat["int8"]["file_mb"]
        lat["change_measured"] = latency.intervals_disjoint(
            lat["fp32"]["p50_by_round_ms"], lat["int8"]["p50_by_round_ms"]
        )
        lat["arithmetic_intensity_B1"] = {
            "fp32_b4": export.arithmetic_intensity(paths["fp32"], 4),
            "int8_b1": export.arithmetic_intensity(paths["fp32"], 1),
        }
        res["step5_latency"] = lat
        print(
            f"П.5: p50 FP32 {lat['fp32']['p50_ms']:.3f} мс, INT8 {lat['int8']['p50_ms']:.3f} мс, "
            f"прискорення {lat['speedup_p50']:.3f}, виміряна зміна: {lat['change_measured']}"
        )
        save()

    if "diag" in steps:
        # діагностика поза умовою: per-channel із reduce_range=True, решта як в основній квантизації
        x_cal = calib_images()
        ensure_prep()
        quantize.quantize(paths["prep"], paths["int8_reduce_range"], x_cal, per_channel=True, reduce_range=True)
        p_rr = quality.onnx_logits(quality.make_session(paths["int8_reduce_range"]), x_test).argmax(1)
        base, rr = quantize.int_ranges(paths["int8"]), quantize.int_ranges(paths["int8_reduce_range"])
        common = [k for k in base["activation_scales"] if k in rr["activation_scales"]]
        same = sum(bool(np.isclose(base["activation_scales"][k], rr["activation_scales"][k])) for k in common)
        per_tensor_q = (
            quantize.int_ranges(paths["int8_per_tensor"])["max_abs_q_weight"]
            if paths["int8_per_tensor"].exists()
            else None
        )
        res["diagnostic_int8_reduce_range"] = {
            **quality.row(p_rr, fp32_pred(), y),
            "int_ranges": {
                "max_abs_q_weight_int8": base["max_abs_q_weight"],
                "max_abs_q_weight_reduce_range": rr["max_abs_q_weight"],
                "max_abs_q_weight_per_tensor": per_tensor_q,
                "activation_scales_equal": same,
                "activation_scales_total": len(common),
                "max_pair_sum_u8xs8_int8": 255 * base["max_abs_q_weight"] * 2,
                "max_pair_sum_u8xs8_reduce_range": 255 * rr["max_abs_q_weight"] * 2,
                "int16_max": 32767,
            },
        }
        res["diagnostic_ort_optimized_graph_ops"] = {
            k: quality.optimized_op_counts(paths[k], out / f"lab4_{k}_ort_optimized.onnx") for k in ["fp32", "int8"]
        }
        r = res["diagnostic_int8_reduce_range"]
        print(f"Діагностика reduce_range: згода {r['agreement']:.4f}, b={r['b']} c={r['c']} — {r['verdict']}")
        save()

    print("Результати:", res_path)


if __name__ == "__main__":
    main()
