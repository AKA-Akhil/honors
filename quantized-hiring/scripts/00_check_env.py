"""Print environment info: OS, Python, GPU, free disk, and whether llama-cpp-python offloads to GPU."""
import sys, platform, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
try:
    print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip())
except Exception as e:
    print("GPU: nvidia-smi unavailable", e)
print("Free disk (GB): %.1f" % (shutil.disk_usage(C.ROOT).free / 1e9))
C.add_cuda_dll_dirs()
import llama_cpp
print("llama-cpp-python:", llama_cpp.__version__, "| supports GPU offload:", llama_cpp.llama_supports_gpu_offload())
