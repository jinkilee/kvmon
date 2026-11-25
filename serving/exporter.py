"""GPU memory breakdown exporter for vLLM.

Exposes, per detected GPU (all GPUs NVML can see):
  - total / used memory                   (NVML)
  - model weight memory                   (parsed from vLLM log: "Model loading took X GiB")
  - KV cache memory, allocated / in use   (log: "Available KV cache memory: X GiB" x vllm:kv_cache_usage_perc)
  - other memory = used - model - kv allocated
  - GPU utilization, memory-controller (read/write) utilization, PCIe RX/TX throughput

GPUs that vLLM does not use (index >= TP*PP) report model=0 and kv=0, so all of
their used memory shows up as "other".
"""

import math
import os
import re
import time
import urllib.request

import pynvml
from prometheus_client import Gauge, start_http_server

LOG_PATH = os.environ.get("VLLM_LOG_PATH", "/var/log/vllm/vllm.log")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9400"))
INTERVAL = float(os.environ.get("EXPORTER_INTERVAL", "1.0"))
TP = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
PP = int(os.environ.get("PIPELINE_PARALLEL_SIZE", "1"))
WORLD = TP * PP
GIB = 1024 ** 3

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MODEL_RE = re.compile(r"Model loading took ([\d.]+) ?(GiB|GB)")
KV_RE = re.compile(r"Available KV cache memory: ([\d.]+) ?(GiB|GB)")
# Worker prefixes differ between vLLM versions:
#   "(VllmWorker rank=1 pid=..)", "(Worker_TP1 pid=..)", "(Worker_PP1_TP0 pid=..)"
RANK_RE = re.compile(r"rank=(\d+)")
PP_TP_RE = re.compile(r"Worker_PP(\d+)_TP(\d+)")
TP_RE = re.compile(r"Worker_TP(\d+)")

LABELS = ["gpu", "name", "uuid"]
P = "kvmon_gpu_"
g_total = Gauge(P + "memory_total_bytes", "Total GPU memory", LABELS)
g_used = Gauge(P + "memory_used_bytes", "Used GPU memory", LABELS)
g_model = Gauge(P + "memory_model_bytes", "Memory taken by model weights", LABELS)
g_kv_alloc = Gauge(P + "memory_kv_cache_allocated_bytes", "Memory reserved for the KV cache", LABELS)
g_kv_used = Gauge(P + "memory_kv_cache_used_bytes", "KV cache memory holding live blocks", LABELS)
g_other = Gauge(P + "memory_other_bytes", "Used memory that is neither weights nor KV cache", LABELS)
g_util = Gauge(P + "utilization_percent", "GPU (SM) utilization", LABELS)
g_mem_util = Gauge(P + "memory_rw_utilization_percent",
                   "Percent of time device memory was being read or written", LABELS)
g_pcie_rx = Gauge(P + "pcie_rx_bytes_per_second", "PCIe RX (host->GPU) throughput", LABELS)
g_pcie_tx = Gauge(P + "pcie_tx_bytes_per_second", "PCIe TX (GPU->host) throughput", LABELS)
g_vllm = Gauge(P + "vllm_assigned", "1 if vLLM uses this GPU", LABELS)
g_received = Gauge("kvmon_vllm_requests_received_total",
                   "Requests accepted by the vLLM engine (finished + running + waiting)")


class LogState:
    """Tails the vLLM log and keeps per-rank model / KV cache sizes in bytes."""

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.inode = None
        self.model = {}
        self.kv = {}

    @staticmethod
    def _rank(line):
        m = PP_TP_RE.search(line)
        if m:
            return int(m.group(1)) * TP + int(m.group(2))
        m = TP_RE.search(line) or RANK_RE.search(line)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _bytes(value, unit):
        return float(value) * (GIB if unit == "GiB" else 1e9)

    def poll(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return
        if st.st_ino != self.inode or st.st_size < self.pos:
            # new or truncated log -> vLLM restarted, forget old numbers
            self.inode, self.pos, self.model, self.kv = st.st_ino, 0, {}, {}
        with open(self.path, "r", errors="replace") as f:
            f.seek(self.pos)
            for line in f:
                line = ANSI_RE.sub("", line)
                m = MODEL_RE.search(line)
                if m:
                    self.model[self._rank(line)] = self._bytes(*m.groups())
                m = KV_RE.search(line)
                if m:
                    self.kv[self._rank(line)] = self._bytes(*m.groups())
            self.pos = f.tell()

    @staticmethod
    def _for_rank(values, rank):
        if rank >= WORLD or not values:
            return 0.0
        if rank in values:
            return values[rank]
        # Some versions only log on rank 0; shards are symmetric, so reuse it.
        return values.get(0, next(iter(values.values())))

    def model_bytes(self, rank):
        return self._for_rank(self.model, rank)

    def kv_bytes(self, rank):
        return self._for_rank(self.kv, rank)


def scrape_vllm():
    """Read vLLM's own /metrics. Returns (kv_usage_ratio, requests_received) or None."""
    try:
        with urllib.request.urlopen(VLLM_METRICS_URL, timeout=1) as r:
            text = r.read().decode()
    except Exception:
        return None
    sums, kv = {}, {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        if name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            kv.setdefault(name, []).append(value)
        elif name in ("vllm:request_success_total", "vllm:num_requests_running",
                      "vllm:num_requests_waiting"):
            sums[name] = sums.get(name, 0.0) + value
    usage = 0.0
    for name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        if kv.get(name):
            v = max(kv[name])
            usage = v if v <= 1.0 else v / 100.0
            break
    # vLLM has no "request arrived" counter. Requests are conserved, so
    # everything the engine ever accepted = finished (any reason) + in flight.
    received = sum(sums.values())
    return usage, received


def nvml(fn, *args, default=math.nan):
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return default


def main():
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    gpus = []
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        name = name.decode() if isinstance(name, bytes) else name
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        gpus.append((i, h, (str(i), name, uuid)))
    print(f"[exporter] {count} GPU(s) detected, vLLM world size={WORLD}, "
          f"serving on :{EXPORTER_PORT}", flush=True)

    logs = LogState(LOG_PATH)
    start_http_server(EXPORTER_PORT)

    pcie_ok = True
    while True:
        t0 = time.time()
        logs.poll()
        scraped = scrape_vllm()
        usage = 0.0
        if scraped:
            usage, received = scraped
            g_received.set(received)
        for i, h, labels in gpus:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            model = logs.model_bytes(i)
            kv = logs.kv_bytes(i)
            g_total.labels(*labels).set(mem.total)
            g_used.labels(*labels).set(mem.used)
            g_model.labels(*labels).set(model)
            g_kv_alloc.labels(*labels).set(kv)
            g_kv_used.labels(*labels).set(kv * usage)
            g_other.labels(*labels).set(max(mem.used - model - kv, 0))
            g_vllm.labels(*labels).set(1 if i < WORLD else 0)

            util = nvml(pynvml.nvmlDeviceGetUtilizationRates, h, default=None)
            g_util.labels(*labels).set(util.gpu if util else math.nan)
            g_mem_util.labels(*labels).set(util.memory if util else math.nan)

            if pcie_ok:
                # NVML returns KB/s sampled over ~20ms; unsupported on some platforms (e.g. WSL)
                rx = nvml(pynvml.nvmlDeviceGetPcieThroughput, h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                tx = nvml(pynvml.nvmlDeviceGetPcieThroughput, h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                if math.isnan(rx) and math.isnan(tx) and i == 0:
                    pcie_ok = False
                    print("[exporter] PCIe throughput not supported, disabling", flush=True)
                g_pcie_rx.labels(*labels).set(rx * 1024)
                g_pcie_tx.labels(*labels).set(tx * 1024)
        time.sleep(max(INTERVAL - (time.time() - t0), 0.05))


if __name__ == "__main__":
    main()
