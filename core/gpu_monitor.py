# -*- coding: utf-8 -*-
"""
GPU Monitor - Real-time GPU status detection and monitoring
Supports NVIDIA (pynvml/nvidia-smi), AMD (rocm-smi), and Apple Silicon (Metal)
"""

import subprocess
import platform
import threading
import time
import os
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict

@dataclass
class GPUInfo:
    """GPU device information snapshot"""
    index: int = 0
    name: str = "Unknown"
    driver_version: str = "N/A"
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    free_memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    temperature: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    fan_speed: float = 0.0
    clock_gpu_mhz: float = 0.0
    clock_mem_mhz: float = 0.0
    pcie_gen: int = 0
    compute_capability: str = "N/A"
    is_available: bool = False
    backend: str = "none"  # cuda, rocm, metal, vulkan, cpu

    @property
    def memory_used_gb(self) -> float:
        return self.used_memory_mb / 1024.0

    @property
    def memory_total_gb(self) -> float:
        return self.total_memory_mb / 1024.0

    @property
    def memory_free_gb(self) -> float:
        return self.free_memory_mb / 1024.0

    @property
    def status_text(self) -> str:
        if not self.is_available:
            return "OFFLINE"
        if self.gpu_utilization > 90:
            return "HEAVY LOAD"
        if self.gpu_utilization > 50:
            return "ACTIVE"
        if self.gpu_utilization > 10:
            return "IDLE"
        return "STANDBY"

    @property
    def health_color(self) -> str:
        """Returns color for status indicators"""
        if not self.is_available:
            return "#FF0000"
        if self.temperature > 85:
            return "#FF4444"
        if self.temperature > 70:
            return "#FFaa00"
        if self.gpu_utilization > 90:
            return "#FFaa00"
        return "#00FF88"


@dataclass
class SystemInfo:
    """System resource information"""
    cpu_percent: float = 0.0
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    os_name: str = ""
    python_version: str = ""


class GPUMonitor:
    """
    Industrial-grade GPU monitor with multiple backend support.
    Polls GPU stats at configurable intervals.
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._gpu_infos: List[GPUInfo] = []
        self._system_info = SystemInfo()
        self._lock = threading.Lock()
        self._backend = "none"
        self._force_gpu = False
        self._pynvml_available = False
        self._initialized = False

        self._detect_backend()

    def _detect_backend(self):
        """Detect available GPU backend"""
        # Try NVIDIA first
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._backend = "cuda"
                self._pynvml_available = True
                self._initialized = True
                pynvml.nvmlShutdown()
                return
        except Exception:
            pass

        # Try nvidia-smi fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                self._backend = "cuda"
                self._initialized = True
                return
        except Exception:
            pass

        # Try AMD ROCm
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._backend = "rocm"
                self._initialized = True
                return
        except Exception:
            pass

        # Try Apple Silicon Metal
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self._backend = "metal"
            self._initialized = True
            return

        # Check if torch CUDA available
        try:
            import torch
            if torch.cuda.is_available():
                self._backend = "cuda"
                self._initialized = True
                return
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._backend = "metal"
                self._initialized = True
                return
        except ImportError:
            pass

        # Check llama-cpp-python GPU support
        try:
            from llama_cpp import Llama
            # If we get here, llama_cpp is available
            # Check for CUDA/Metal support flags
            self._backend = "llama_cpp"
            self._initialized = True
            return
        except ImportError:
            pass

        self._backend = "cpu"
        self._initialized = True

    def _poll_nvidia_pynvml(self) -> List[GPUInfo]:
        """Poll NVIDIA GPUs via pynvml"""
        gpus = []
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()

            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = GPUInfo(index=i, is_available=True, backend="cuda")

                try:
                    info.name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(info.name, bytes):
                        info.name = info.name.decode('utf-8')
                except Exception:
                    pass

                try:
                    info.driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(info.driver_version, bytes):
                        info.driver_version = info.driver_version.decode('utf-8')
                except Exception:
                    pass

                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info.total_memory_mb = mem.total / (1024 * 1024)
                    info.used_memory_mb = mem.used / (1024 * 1024)
                    info.free_memory_mb = mem.free / (1024 * 1024)
                    info.memory_percent = (mem.used / mem.total) * 100 if mem.total > 0 else 0
                except Exception:
                    pass

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    info.gpu_utilization = util.gpu
                except Exception:
                    pass

                try:
                    info.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass

                try:
                    info.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass

                try:
                    info.power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:
                    pass

                try:
                    info.fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    pass

                try:
                    info.clock_gpu_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                except Exception:
                    pass

                try:
                    info.clock_mem_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    pass

                gpus.append(info)

            pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpus

    def _poll_nvidia_smi(self) -> List[GPUInfo]:
        """Fallback: poll via nvidia-smi CLI"""
        gpus = []
        try:
            fields = "index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,clocks.current.graphics,clocks.current.memory"
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 13:
                        info = GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            driver_version=parts[2],
                            total_memory_mb=float(parts[3]),
                            used_memory_mb=float(parts[4]),
                            free_memory_mb=float(parts[5]),
                            gpu_utilization=float(parts[6]) if parts[6] != '[N/A]' else 0,
                            temperature=float(parts[7]) if parts[7] != '[N/A]' else 0,
                            power_draw_w=float(parts[8]) if parts[8] != '[N/A]' else 0,
                            power_limit_w=float(parts[9]) if parts[9] != '[N/A]' else 0,
                            fan_speed=float(parts[10]) if parts[10] != '[N/A]' else 0,
                            clock_gpu_mhz=float(parts[11]) if parts[11] != '[N/A]' else 0,
                            clock_mem_mhz=float(parts[12]) if parts[12] != '[N/A]' else 0,
                            is_available=True,
                            backend="cuda"
                        )
                        info.memory_percent = (info.used_memory_mb / info.total_memory_mb * 100) if info.total_memory_mb > 0 else 0
                        gpus.append(info)
        except Exception:
            pass
        return gpus

    def _poll_system_info(self) -> SystemInfo:
        """Poll system CPU/RAM info"""
        info = SystemInfo()
        info.os_name = f"{platform.system()} {platform.release()}"
        info.cpu_count = os.cpu_count() or 1

        try:
            import psutil
            info.cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            info.ram_total_gb = mem.total / (1024 ** 3)
            info.ram_used_gb = mem.used / (1024 ** 3)
            info.ram_percent = mem.percent
            freqs = psutil.cpu_freq()
            if freqs:
                info.cpu_freq_mhz = freqs.current
        except ImportError:
            # Fallback without psutil
            try:
                if platform.system() == "Linux":
                    with open("/proc/meminfo") as f:
                        lines = f.readlines()
                    for l in lines:
                        if l.startswith("MemTotal:"):
                            info.ram_total_gb = int(l.split()[1]) / (1024 * 1024)
                        elif l.startswith("MemAvailable:"):
                            avail = int(l.split()[1]) / (1024 * 1024)
                            info.ram_used_gb = info.ram_total_gb - avail
                            info.ram_percent = (info.ram_used_gb / info.ram_total_gb * 100) if info.ram_total_gb > 0 else 0
            except Exception:
                pass

        return info

    def poll_once(self) -> tuple:
        """Single poll of all GPU and system stats"""
        gpus = []

        if self._backend == "cuda":
            if self._pynvml_available:
                gpus = self._poll_nvidia_pynvml()
            if not gpus:
                gpus = self._poll_nvidia_smi()

        if not gpus and self._backend == "metal":
            # Apple Silicon - limited info
            gpu = GPUInfo(
                index=0,
                name=f"Apple {platform.processor()} (Metal)",
                is_available=True,
                backend="metal"
            )
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    total_bytes = int(result.stdout.strip())
                    gpu.total_memory_mb = total_bytes / (1024 * 1024)  # Unified memory
            except Exception:
                pass
            gpus = [gpu]

        if not gpus:
            gpus = [GPUInfo(
                index=0,
                name="No GPU Detected (CPU Only)",
                is_available=False,
                backend="cpu"
            )]

        sys_info = self._poll_system_info()

        with self._lock:
            self._gpu_infos = gpus
            self._system_info = sys_info

        return gpus, sys_info

    def start(self):
        """Start background polling"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _poll_loop(self):
        while self._running:
            gpus, sys_info = self.poll_once()
            for cb in self._callbacks:
                try:
                    cb(gpus, sys_info)
                except Exception:
                    pass
            time.sleep(self.poll_interval)

    def add_callback(self, cb: Callable):
        self._callbacks.append(cb)

    def remove_callback(self, cb: Callable):
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    @property
    def gpu_infos(self) -> List[GPUInfo]:
        with self._lock:
            return list(self._gpu_infos)

    @property
    def system_info(self) -> SystemInfo:
        with self._lock:
            return self._system_info

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def has_gpu(self) -> bool:
        return self._backend not in ("cpu", "none")

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        infos = self.gpu_infos
        return infos[0] if infos and infos[0].is_available else None

    def get_recommended_gpu_layers(self, model_size_gb: float = 4.0) -> int:
        """Recommend number of GPU layers based on available VRAM"""
        gpu = self.primary_gpu
        if not gpu or not gpu.is_available:
            return 0

        free_gb = gpu.free_memory_mb / 1024.0
        # Rough heuristic: each layer ~= model_size / 33 for typical LLMs
        if free_gb <= 0.5:
            return 0
        estimated_layers = int((free_gb / model_size_gb) * 33)
        return max(0, min(estimated_layers, 999))

    def get_status_summary(self) -> Dict:
        """Get a summary dict for logging/display"""
        gpu = self.primary_gpu
        sys = self.system_info
        return {
            "backend": self._backend,
            "gpu_available": self.has_gpu,
            "gpu_name": gpu.name if gpu else "N/A",
            "gpu_utilization": gpu.gpu_utilization if gpu else 0,
            "vram_used_gb": gpu.memory_used_gb if gpu else 0,
            "vram_total_gb": gpu.memory_total_gb if gpu else 0,
            "vram_percent": gpu.memory_percent if gpu else 0,
            "temperature": gpu.temperature if gpu else 0,
            "cpu_percent": sys.cpu_percent,
            "ram_used_gb": sys.ram_used_gb,
            "ram_total_gb": sys.ram_total_gb,
        }
