"""Central constants for the quantization-hiring audit (round 2)."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
DATA = ROOT / "data"
MODELS_DIR = ROOT / "models"
TOOLS = ROOT / "tools" / "llama.cpp"
RAW = Path(os.environ.get("QH_RAW", ROOT / "results" / "raw"))            # overridable for tests on synthetic data
TABLES = Path(os.environ.get("QH_TABLES", ROOT / "results" / "tables"))
FIGS = Path(os.environ.get("QH_FIGS", ROOT / "results" / "figures"))

# model key -> display name. Full-precision sources are recorded in results/model_provenance.csv
MODELS = {"llama32_3b": "Llama-3.2-3B-Instruct", "qwen25_3b": "Qwen2.5-3B-Instruct", "phi35_mini": "Phi-3.5-mini-instruct"}
# Reference + quantized levels. recipe: ref (F16), imat (importance matrix from WikiText-2 train), noimat (ablation)
REF_LEVEL = "F16"
LEVELS_IMAT = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
LEVELS_NOIMAT = ["Q4_K_M", "Q3_K_M", "Q2_K"]
LEVEL_ORDER = [REF_LEVEL] + LEVELS_IMAT
TEMPLATES = ["A", "B", "C"]
GROUPS = ["WF", "BF", "WM", "BM"]
TIERS = ["strong", "borderline_high", "borderline_low", "weak"]


def run_spec():
    """All (level, recipe, templates) combinations in priority order for one model."""
    spec = [(REF_LEVEL, "ref", TEMPLATES)] + [(l, "imat", TEMPLATES) for l in LEVELS_IMAT]
    spec += [(l, "noimat", ["A"]) for l in LEVELS_NOIMAT]
    return spec


def gguf_path(model, level, recipe):
    if recipe == "ref":
        return MODELS_DIR / model / f"{model}-{REF_LEVEL}.gguf"
    return MODELS_DIR / model / f"{model}-{level}-{recipe}.gguf"


def llama_bin(name, prefer="cuda"):
    """Path to a llama.cpp executable, preferring the CUDA build and falling back to the CPU build."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    for sub in ([prefer, "cpu"] if prefer == "cuda" else ["cpu"]):
        p = TOOLS / sub / exe
        if p.exists():
            return p
    raise FileNotFoundError(exe)


def add_cuda_dll_dirs():
    """Make the pip-installed CUDA runtime DLLs visible to llama-cpp-python (Windows)."""
    if sys.platform != "win32":
        return
    sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cuda_runtime", "cuda_nvrtc"):
        d = sp / sub / "bin"
        if d.exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
