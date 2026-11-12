import platform
import subprocess
import sys
from pathlib import Path

import numpy
import onnx
import onnxruntime
import PIL
import sklearn
import torch
import torchvision

UNKNOWN = "unknown"
X86_INT8_FLAGS = ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8"]
ARM_INT8_SYSCTL = {
    "SDOT (FEAT_DotProd)": "hw.optional.arm.FEAT_DotProd",
    "I8MM (FEAT_I8MM)": "hw.optional.arm.FEAT_I8MM",
}


def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def linux_cpu():
    model, flags = None, None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, val = line.partition(":")
            key = key.strip()
            if key == "model name" and model is None:
                model = val.strip()
            elif key in ("flags", "Features") and flags is None:
                flags = set(val.split())
    except OSError:
        pass
    if model is None:
        for line in (run_cmd(["lscpu"]) or "").splitlines():
            if line.startswith("Model name:"):
                model = line.split(":", 1)[1].strip()
                break
    return model, flags


def cpu_info():
    system, machine = platform.system(), platform.machine()
    model, isa = None, {}
    if system == "Linux":
        model, flags = linux_cpu()
        if flags is not None and machine in ("x86_64", "AMD64"):
            isa = {f: f in flags for f in X86_INT8_FLAGS}
        elif flags is not None:
            isa = {"asimddp (SDOT)": "asimddp" in flags, "i8mm": "i8mm" in flags}
    elif system == "Darwin":
        model = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        for name, key in ARM_INT8_SYSCTL.items():
            value = run_cmd(["sysctl", "-n", key])
            isa[name] = (value == "1") if value is not None else UNKNOWN
    elif system == "Windows":
        model = platform.processor() or None
    return {"cpu_model": model or UNKNOWN, "machine": machine, "int8_isa_flags": isa or UNKNOWN}


def os_info():
    system, kernel = platform.system(), platform.release()
    name = None
    if system == "Linux":
        try:
            lines = Path("/etc/os-release").read_text().splitlines()
            name = dict(line.split("=", 1) for line in lines if "=" in line).get("PRETTY_NAME", "").strip('"')
        except OSError:
            pass
    elif system == "Darwin":
        version = platform.mac_ver()[0]
        name = f"macOS {version}" if version else None
    elif system == "Windows":
        name = platform.platform()
    return {
        "system": system,
        "os": name or platform.platform() or UNKNOWN,
        "kernel": kernel,
        "wsl2": system == "Linux" and "microsoft-standard-wsl2" in kernel.lower(),
    }


def collect():
    return {
        **cpu_info(),
        **os_info(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "latency_batch_size": 1,
        "quality_batch_size": 64,
    }
