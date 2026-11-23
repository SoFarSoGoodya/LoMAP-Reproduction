"""Re-run the Phase C wall-crossing evaluator with an FP32 GPU IVF-Flat index."""

import os
from pathlib import Path
import runpy
import sys
import time

import faiss


class GpuFaiss:
    def __init__(self, bank, k, nlist):
        start = time.perf_counter()
        self.cpu = self.cpu_factory(bank, k, nlist)
        self.cpu_index = self.cpu.index
        self.resources = faiss.StandardGpuResources()
        self.resources.setTempMemory(512 * 1024**2)
        self.index = faiss.index_cpu_to_gpu(self.resources, 0, self.cpu_index)
        self.index.nprobe = self.cpu_index.nprobe
        self.cpu.index = self.index
        del self.cpu_index
        print(f"GPU FP32 IVF-Flat ready: {len(bank)} windows in {time.perf_counter() - start:.2f}s", flush=True)

    @property
    def search_calls(self):
        return self.cpu.search_calls

    @search_calls.setter
    def search_calls(self, value):
        self.cpu.search_calls = value

    def search(self, queries, k):
        return self.cpu.search(queries, k)

    def get_original_vectors(self, indices):
        return self.cpu.get_original_vectors(indices)


def main():
    phase_b = Path(os.environ.get("LOMAP_PHASE_B", Path(__file__).resolve().parents[1] / "phase_b"))
    sys.path.insert(0, str(phase_b))
    import smoke

    GpuFaiss.cpu_factory = smoke.CpuFaiss
    smoke.CpuFaiss = GpuFaiss
    runpy.run_path(str(phase_b / "medium_evaluate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
