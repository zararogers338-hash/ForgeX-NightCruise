# -*- coding: utf-8 -*-
"""
Night Cruise v7.1 Industrial — Full-Featured Training Platform
=============================================================
Features:
  - Real-time GPU monitoring dashboard with gauges
  - Auto GPU detection and health reporting
  - Force GPU inference button
  - GPU layer count control (n_gpu_layers)
  - HuggingFace Transformers backend support
  - GGUF model support via llama-cpp-python
  - Adjustable token limits, top_p, top_k, min_p, repeat_penalty
  - Context window size control
  - Batch size control
  - Performance metrics (tok/s, latency)
  - Industrial dark theme with status indicators
  - Dual-model generation + validator
  - Advanced generation modes (attack, hallucination defense, deep thinking, quick)
  - Quality guard with toggle
  - Multi-format export (JSONL, JSON, TXT)
  - Discipline classification charts
  - Session config save/load
  - Model cache management with VRAM cleanup

v7.1 Fresh Session Optimization:
  - ★ 每次流式生成都开启全新的 API 会话窗口
  - ★ 自动清除 Ollama/GGUF/HF 的 KV Cache
  - ★ 防止 t/s 随处理文件数增加而持续下降
  - ★ Ollama 使用 /api/chat + keep_alive=0 强制刷新上下文
  - ★ 每个文件处理间隔自动 reset_context()
"""

from __future__ import annotations

import json
import re
import threading
import os
import sys
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter

import customtkinter as ctk
from customtkinter import (CTkTabview, CTkTextbox, CTkComboBox, CTkButton,
                           CTkEntry, CTkCheckBox, CTkLabel, CTkFrame,
                           CTkScrollableFrame, CTkSlider, CTkProgressBar,
                           CTkSwitch, CTkOptionMenu, CTkSegmentedButton)
from tkinter import filedialog, messagebox
import tkinter as tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Local imports
from core.gpu_monitor import GPUMonitor, GPUInfo, SystemInfo
from core.inference import InferenceEngine, InferenceConfig, GenerationConfig, get_models, InferenceMetrics
from core.text_processing import (extract_text, heuristic_abstract, classify_paper,
                                   quality_guard, parse_validator_output, SUPPORTED_EXTS)
from core.prompts import (DEFAULT_SYSTEM_PROMPT, VALIDATOR_PROMPT, VERSION, APP_TITLE,
                           DEFAULT_OLLAMA_URL, DEFAULT_CUSTOM_URL, DEFAULT_MODEL,
                           DEFAULT_MAX_CHARS, DEFAULT_NUM_PREDICT, DEFAULT_OUTPUT_PREFIX,
                           LOW_CONF_THRESHOLD)

# ====================== Industrial Color Scheme ======================
COLORS = {
    "bg_dark": "#0a0e14",
    "bg_panel": "#111820",
    "bg_card": "#161d27",
    "bg_input": "#1a222e",
    "border": "#2a3545",
    "accent_blue": "#00b4d8",
    "accent_green": "#00e676",
    "accent_red": "#ff1744",
    "accent_orange": "#ff9100",
    "accent_yellow": "#ffd600",
    "accent_purple": "#bb86fc",
    "text_primary": "#e8eaed",
    "text_secondary": "#8a9ab5",
    "text_dim": "#4a5568",
    "gauge_bg": "#1e2a3a",
    "gauge_fill": "#00b4d8",
    "warning_bg": "#2d1f00",
    "error_bg": "#2d0000",
    "success_bg": "#002d0a",
}


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ====================== Industrial Gauge Widget ======================
class GaugeWidget(CTkFrame):
    """Circular gauge for displaying percentages"""

    def __init__(self, master, title="", unit="%", max_val=100, size=120, **kwargs):
        super().__init__(master, **kwargs)
        self.title = title
        self.unit = unit
        self.max_val = max_val
        self.size = size
        self._value = 0

        self.configure(fg_color=COLORS["bg_card"], corner_radius=8)

        self.canvas = tk.Canvas(
            self, width=size, height=size,
            bg=COLORS["bg_card"], highlightthickness=0, bd=0
        )
        self.canvas.pack(padx=5, pady=(5, 0))

        self.label_title = CTkLabel(self, text=title, text_color=COLORS["text_secondary"],
                                     font=("Consolas", 10))
        self.label_title.pack(pady=(0, 2))

        self.label_value = CTkLabel(self, text="0%", text_color=COLORS["text_primary"],
                                     font=("Consolas", 14, "bold"))
        self.label_value.pack(pady=(0, 5))

        self._draw_gauge(0)

    def _draw_gauge(self, value):
        self.canvas.delete("all")
        cx, cy = self.size // 2, self.size // 2
        r = self.size // 2 - 10
        r_inner = r - 8

        # Background arc
        self.canvas.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=225, extent=-270, style="arc",
            outline=COLORS["gauge_bg"], width=8
        )

        # Value arc
        pct = min(value / self.max_val, 1.0) if self.max_val > 0 else 0
        extent = -270 * pct

        # Color based on value
        if pct > 0.9:
            color = COLORS["accent_red"]
        elif pct > 0.7:
            color = COLORS["accent_orange"]
        elif pct > 0.4:
            color = COLORS["accent_yellow"]
        else:
            color = COLORS["accent_green"]

        if extent != 0:
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=225, extent=extent, style="arc",
                outline=color, width=8
            )

    def set_value(self, value, text=None):
        self._value = value
        self._draw_gauge(value)
        if text is None:
            if self.unit == "%":
                text = f"{value:.1f}%"
            elif self.unit == "°C":
                text = f"{value:.0f}°C"
            elif self.unit == "W":
                text = f"{value:.0f}W"
            elif self.unit == "GB":
                text = f"{value:.1f}GB"
            elif self.unit == "MHz":
                text = f"{value:.0f}"
            else:
                text = f"{value:.1f}{self.unit}"
        self.label_value.configure(text=text)


# ====================== Status Indicator Widget ======================
class StatusIndicator(CTkFrame):
    """LED-style status indicator"""

    def __init__(self, master, label="Status", **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")

        self._canvas = tk.Canvas(self, width=14, height=14,
                                  bg=COLORS["bg_card"], highlightthickness=0)
        self._canvas.pack(side="left", padx=(0, 6))
        self._dot = self._canvas.create_oval(2, 2, 12, 12, fill="#555555", outline="")

        self._label = CTkLabel(self, text=label, text_color=COLORS["text_secondary"],
                                font=("Consolas", 11))
        self._label.pack(side="left")

    def set_status(self, color: str, text: str = None):
        self._canvas.itemconfig(self._dot, fill=color)
        if text:
            self._label.configure(text=text)


# ====================== GPU Dashboard Panel ======================
class GPUDashboard(CTkFrame):
    """Real-time GPU monitoring dashboard"""

    def __init__(self, master, gpu_monitor: GPUMonitor, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLORS["bg_panel"], corner_radius=10)
        self.gpu_monitor = gpu_monitor

        # Header
        header = CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 5))

        CTkLabel(header, text="⚡ GPU CONTROL CENTER",
                 font=("Consolas", 16, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(side="left")

        self.gpu_status = StatusIndicator(header, label="DETECTING...")
        self.gpu_status.pack(side="right", padx=10)

        # GPU Info Bar
        self.info_bar = CTkLabel(
            self, text="Scanning hardware...",
            font=("Consolas", 11), text_color=COLORS["text_secondary"],
            fg_color=COLORS["bg_card"], corner_radius=6, anchor="w"
        )
        self.info_bar.pack(fill="x", padx=15, pady=5, ipady=4)

        # Gauges Row
        gauges_frame = CTkFrame(self, fg_color="transparent")
        gauges_frame.pack(fill="x", padx=10, pady=5)

        self.gauge_gpu_util = GaugeWidget(gauges_frame, title="GPU Load", unit="%", size=100)
        self.gauge_gpu_util.pack(side="left", padx=5, expand=True)

        self.gauge_vram = GaugeWidget(gauges_frame, title="VRAM", unit="%", size=100)
        self.gauge_vram.pack(side="left", padx=5, expand=True)

        self.gauge_temp = GaugeWidget(gauges_frame, title="Temp", unit="°C", max_val=100, size=100)
        self.gauge_temp.pack(side="left", padx=5, expand=True)

        self.gauge_power = GaugeWidget(gauges_frame, title="Power", unit="W", max_val=350, size=100)
        self.gauge_power.pack(side="left", padx=5, expand=True)

        self.gauge_cpu = GaugeWidget(gauges_frame, title="CPU", unit="%", size=100)
        self.gauge_cpu.pack(side="left", padx=5, expand=True)

        self.gauge_ram = GaugeWidget(gauges_frame, title="RAM", unit="%", size=100)
        self.gauge_ram.pack(side="left", padx=5, expand=True)

        # VRAM Detail Bar
        self.vram_detail = CTkLabel(
            self, text="VRAM: -- / -- GB  |  Free: -- GB",
            font=("Consolas", 10), text_color=COLORS["text_dim"]
        )
        self.vram_detail.pack(fill="x", padx=15, pady=(0, 2))

        # Performance Metrics Bar
        self.perf_bar = CTkLabel(
            self, text="Clock: -- MHz  |  Mem Clock: -- MHz  |  Fan: --%  |  Driver: --",
            font=("Consolas", 10), text_color=COLORS["text_dim"]
        )
        self.perf_bar.pack(fill="x", padx=15, pady=(0, 5))

        # Warning banner (hidden by default)
        self.warning_frame = CTkFrame(self, fg_color=COLORS["warning_bg"], corner_radius=6)
        self.warning_label = CTkLabel(
            self.warning_frame,
            text="⚠️  WARNING: No GPU detected — running on CPU. Performance will be severely degraded!",
            font=("Consolas", 12, "bold"), text_color=COLORS["accent_orange"]
        )
        self.warning_label.pack(padx=15, pady=8)
        # Don't pack warning_frame yet — shown conditionally

        self._cpu_warning_shown = False

    def update_display(self, gpus: List[GPUInfo], sys_info: SystemInfo):
        """Update all dashboard elements"""
        gpu = gpus[0] if gpus else None

        if gpu and gpu.is_available:
            self.gpu_status.set_status(gpu.health_color, f"{gpu.status_text}")
            self.info_bar.configure(
                text=f"  🎮 {gpu.name}  |  Backend: {gpu.backend.upper()}  |  Driver: {gpu.driver_version}"
            )
            self.gauge_gpu_util.set_value(gpu.gpu_utilization)
            self.gauge_vram.set_value(gpu.memory_percent)
            self.gauge_temp.set_value(gpu.temperature)
            self.gauge_power.set_value(gpu.power_draw_w)
            self.gauge_power.max_val = max(gpu.power_limit_w, 1)

            self.vram_detail.configure(
                text=f"VRAM: {gpu.memory_used_gb:.1f} / {gpu.memory_total_gb:.1f} GB  |  "
                     f"Free: {gpu.memory_free_gb:.1f} GB  |  Usage: {gpu.memory_percent:.1f}%"
            )
            self.perf_bar.configure(
                text=f"Clock: {gpu.clock_gpu_mhz:.0f} MHz  |  Mem Clock: {gpu.clock_mem_mhz:.0f} MHz  |  "
                     f"Fan: {gpu.fan_speed:.0f}%  |  PCIe Gen{gpu.pcie_gen}"
            )

            if self._cpu_warning_shown:
                self.warning_frame.pack_forget()
                self._cpu_warning_shown = False
        else:
            self.gpu_status.set_status(COLORS["accent_red"], "NO GPU")
            self.info_bar.configure(text="  ❌ No GPU detected — CPU inference mode")
            self.gauge_gpu_util.set_value(0, text="N/A")
            self.gauge_vram.set_value(0, text="N/A")
            self.gauge_temp.set_value(0, text="N/A")
            self.gauge_power.set_value(0, text="N/A")

            if not self._cpu_warning_shown:
                self.warning_frame.pack(fill="x", padx=15, pady=5)
                self._cpu_warning_shown = True

        # System gauges
        self.gauge_cpu.set_value(sys_info.cpu_percent)
        self.gauge_ram.set_value(sys_info.ram_percent)


# ====================== Performance Metrics Panel ======================
class MetricsPanel(CTkFrame):
    """Display inference performance metrics"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLORS["bg_card"], corner_radius=8)

        CTkLabel(self, text="📊 INFERENCE METRICS",
                 font=("Consolas", 12, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(anchor="w", padx=10, pady=(8, 4))

        self.metrics_text = CTkLabel(
            self, text="Tokens: --  |  Speed: -- tok/s  |  Session: --  |  Latency: --s  |  Backend: --",
            font=("Consolas", 11), text_color=COLORS["text_secondary"],
            anchor="w"
        )
        self.metrics_text.pack(fill="x", padx=10, pady=(0, 8))

    def update_metrics(self, m: InferenceMetrics):
        self.metrics_text.configure(
            text=f"Tokens: {m.tokens_generated}  |  "
                 f"Speed: {m.tokens_per_second:.1f} tok/s  |  "
                 f"Session: #{m.session_id}  |  "
                 f"Time: {m.total_time_s:.1f}s  |  "
                 f"Backend: {m.backend_used}"
        )


# ====================== Main Application ======================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.geometry("1700x1050")
        self.minsize(1400, 900)
        self.configure(fg_color=COLORS["bg_dark"])
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ---- GPU Monitor ----
        self.gpu_monitor = GPUMonitor(poll_interval=1.5)
        self.gpu_monitor.poll_once()  # Initial poll

        # ---- Variables ----
        self._init_variables()

        # ---- Build UI ----
        self._build_ui()
        self.update_provider()
        self.refresh_models()

        # ---- Start GPU monitor ----
        self.gpu_monitor.add_callback(self._on_gpu_update)
        self.gpu_monitor.start()

        # Bindings
        self.enable_dual.trace_add("write", self.on_dual_toggle)
        self.enable_validator.trace_add("write", self.on_validator_toggle)

        self._log_startup_info()

    def _init_variables(self):
        """Initialize all control variables"""
        self.paper_dir = ctk.StringVar()
        self.out_dir = ctk.StringVar(value=str(Path.cwd()))
        self.provider = ctk.StringVar(value="ollama")
        self.base_url = ctk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.api_key = ctk.StringVar(value="")
        self.gguf_path = ctk.StringVar(value="")
        self.hf_model_id = ctk.StringVar(value="")
        self.model = ctk.StringVar(value=DEFAULT_MODEL)
        self.second_model = ctk.StringVar(value="")
        self.validator_gguf = ctk.StringVar(value="")
        self.output_prefix = ctk.StringVar(value=DEFAULT_OUTPUT_PREFIX)
        self.recursive = ctk.BooleanVar(value=True)
        self.max_chars = ctk.IntVar(value=DEFAULT_MAX_CHARS)
        self.num_predict = ctk.IntVar(value=DEFAULT_NUM_PREDICT)
        self.export_format = ctk.StringVar(value="jsonl")
        self.enable_dual = ctk.BooleanVar(value=False)
        self.enable_validator = ctk.BooleanVar(value=True)
        self.enable_quality_guard = ctk.BooleanVar(value=True)

        # Temperature and sampling
        self.temperature = ctk.DoubleVar(value=0.7)
        self.top_p = ctk.DoubleVar(value=0.95)
        self.top_k = ctk.IntVar(value=40)
        self.min_p = ctk.DoubleVar(value=0.05)
        self.repeat_penalty = ctk.DoubleVar(value=1.1)
        self.presence_penalty = ctk.DoubleVar(value=0.0)
        self.frequency_penalty = ctk.DoubleVar(value=0.0)

        # GPU / Engine controls
        self.n_gpu_layers = ctk.IntVar(value=-1)  # -1 = auto
        self.n_ctx = ctk.IntVar(value=8192)
        self.n_batch = ctk.IntVar(value=512)
        self.n_threads = ctk.IntVar(value=0)  # 0 = auto
        self.flash_attention = ctk.BooleanVar(value=True)
        self.use_mmap = ctk.BooleanVar(value=True)
        self.force_gpu = ctk.BooleanVar(value=False)
        self.seed = ctk.IntVar(value=-1)
        self.is_llama_server = ctk.BooleanVar(value=False)  # ★ llama.cpp server 模式

        # Advanced modes
        self.enable_attack = ctk.BooleanVar(value=False)
        self.enable_hallucination_defense = ctk.BooleanVar(value=False)
        self.enable_deep_thinking = ctk.BooleanVar(value=False)
        self.enable_quick_mode = ctk.BooleanVar(value=False)
        self.modes_visible = False

        self.current_system_prompt = DEFAULT_SYSTEM_PROMPT
        self._stop_flag = False
        self.history: List[Dict] = []
        self.major_counter = Counter()
        self.sub_counter = Counter()
        self._inference_engine: Optional[InferenceEngine] = None

    def _build_ui(self):
        """Build the complete industrial UI"""
        # ============ TOP: GPU Dashboard ============
        self.gpu_dashboard = GPUDashboard(self, self.gpu_monitor)
        self.gpu_dashboard.pack(fill="x", padx=10, pady=(8, 4))

        # ============ MAIN: Tabbed Interface ============
        tabview = CTkTabview(self, fg_color=COLORS["bg_panel"], segmented_button_fg_color=COLORS["bg_card"])
        tabview.pack(fill="both", expand=True, padx=10, pady=(4, 4))

        tab_main = tabview.add("🎛️ 主控台")
        tab_engine = tabview.add("⚙️ 推理引擎")
        tab_sampling = tabview.add("🎯 采样参数")
        tab_prompt = tabview.add("📝 提示词")
        tab_preview = tabview.add("👁️ 实时预览")
        tab_history = tabview.add("📜 历史记录")
        tab_log = tabview.add("📋 运行日志")
        tab_chart = tabview.add("📊 学科统计")

        self._build_main_tab(tab_main)
        self._build_engine_tab(tab_engine)
        self._build_sampling_tab(tab_sampling)
        self._build_prompt_tab(tab_prompt)
        self._build_preview_tab(tab_preview)
        self._build_history_tab(tab_history)
        self._build_log_tab(tab_log)
        self._build_chart_tab(tab_chart)

        # ============ BOTTOM: Status Bar ============
        self._build_status_bar()

    # ======================== TAB BUILDERS ========================

    def _build_main_tab(self, parent):
        """Main control panel"""
        scroll = CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        scroll.grid_columnconfigure(1, weight=1)

        row = 0
        # Section: Paths
        self._section_header(scroll, "📁 INPUT / OUTPUT", row); row += 1

        CTkLabel(scroll, text="论文文件夹:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        CTkEntry(scroll, textvariable=self.paper_dir, fg_color=COLORS["bg_input"]).grid(
            row=row, column=1, sticky="ew", padx=5)
        CTkButton(scroll, text="Browse", command=self.pick_paper_dir, width=90,
                  fg_color=COLORS["border"]).grid(row=row, column=2, padx=5)
        row += 1

        CTkLabel(scroll, text="输出目录:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        CTkEntry(scroll, textvariable=self.out_dir, fg_color=COLORS["bg_input"]).grid(
            row=row, column=1, sticky="ew", padx=5)
        CTkButton(scroll, text="Browse", command=self.pick_out_dir, width=90,
                  fg_color=COLORS["border"]).grid(row=row, column=2, padx=5)
        row += 1

        CTkLabel(scroll, text="输出前缀:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        CTkEntry(scroll, textvariable=self.output_prefix, fg_color=COLORS["bg_input"]).grid(
            row=row, column=1, sticky="ew", padx=5)
        row += 1

        # Section: Provider
        self._section_header(scroll, "🔌 MODEL PROVIDER", row); row += 1

        CTkLabel(scroll, text="Provider:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        CTkComboBox(scroll, values=["ollama", "openai_compatible", "local_gguf", "huggingface"],
                    variable=self.provider, command=self.update_provider,
                    fg_color=COLORS["bg_input"], dropdown_fg_color=COLORS["bg_card"]
                    ).grid(row=row, column=1, sticky="ew", padx=5)
        row += 1

        CTkLabel(scroll, text="Base URL:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        self.base_url_entry = CTkEntry(scroll, textvariable=self.base_url, fg_color=COLORS["bg_input"])
        self.base_url_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5)
        row += 1

        CTkLabel(scroll, text="API Key:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        self.api_key_entry = CTkEntry(scroll, textvariable=self.api_key, show="•",
                                       fg_color=COLORS["bg_input"])
        self.api_key_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5)
        row += 1

        CTkLabel(scroll, text="GGUF 文件:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        self.gguf_entry = CTkEntry(scroll, textvariable=self.gguf_path, fg_color=COLORS["bg_input"])
        self.gguf_entry.grid(row=row, column=1, sticky="ew", padx=5)
        self.gguf_button = CTkButton(scroll, text="Select GGUF", command=self.pick_gguf, width=100,
                                      fg_color=COLORS["border"])
        self.gguf_button.grid(row=row, column=2, padx=5)
        row += 1

        CTkLabel(scroll, text="HF Model ID:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        self.hf_entry = CTkEntry(scroll, textvariable=self.hf_model_id, fg_color=COLORS["bg_input"],
                                  placeholder_text="e.g. Qwen/Qwen2.5-1.5B-Instruct")
        self.hf_entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5)
        row += 1

        CTkLabel(scroll, text="主模型:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).grid(row=row, column=0, sticky="w", pady=6, padx=10)
        self.model_combo = CTkComboBox(scroll, variable=self.model, state="readonly",
                                        fg_color=COLORS["bg_input"],
                                        dropdown_fg_color=COLORS["bg_card"])
        self.model_combo.grid(row=row, column=1, sticky="ew", padx=5)

        model_btns = CTkFrame(scroll, fg_color="transparent")
        model_btns.grid(row=row, column=2, padx=5)
        CTkButton(model_btns, text="🔄", command=self.refresh_models, width=36,
                  fg_color=COLORS["border"]).pack(side="left", padx=2)
        CTkButton(model_btns, text="Test", command=self.test_api, width=50,
                  fg_color=COLORS["accent_orange"], text_color="black").pack(side="left", padx=2)
        row += 1

        # Section: Dual Model & Validator
        self._section_header(scroll, "🔗 DUAL MODEL & VALIDATOR", row); row += 1

        dual_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        dual_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        CTkCheckBox(dual_frame, text="启用双模型生成", variable=self.enable_dual,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15, pady=8)
        CTkLabel(dual_frame, text="Second:", text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        self.second_model_combo = CTkComboBox(dual_frame, variable=self.second_model, state="disabled",
                                               fg_color=COLORS["bg_input"],
                                               dropdown_fg_color=COLORS["bg_card"], width=200)
        self.second_model_combo.pack(side="left", padx=5)
        row += 1

        val_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        val_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        CTkCheckBox(val_frame, text="启用验证器 (GGUF, temp=0.0)", variable=self.enable_validator,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15, pady=8)
        CTkLabel(val_frame, text="GGUF:", text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        self.validator_entry = CTkEntry(val_frame, textvariable=self.validator_gguf,
                                         fg_color=COLORS["bg_input"], width=300)
        self.validator_entry.pack(side="left", padx=5)
        CTkButton(val_frame, text="Select", command=self.pick_validator_gguf, width=60,
                  fg_color=COLORS["border"]).pack(side="left", padx=5)
        row += 1

        # Section: Options row
        opts_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        opts_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        CTkCheckBox(opts_frame, text="递归子目录", variable=self.recursive,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15, pady=8)
        CTkCheckBox(opts_frame, text="质量守卫", variable=self.enable_quality_guard,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15)
        CTkLabel(opts_frame, text="Max Chars:", text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        CTkEntry(opts_frame, textvariable=self.max_chars, width=80,
                 fg_color=COLORS["bg_input"]).pack(side="left", padx=5)
        CTkLabel(opts_frame, text="Max Tokens:", text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        CTkEntry(opts_frame, textvariable=self.num_predict, width=80,
                 fg_color=COLORS["bg_input"]).pack(side="left", padx=5)
        CTkLabel(opts_frame, text="Format:", text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        CTkComboBox(opts_frame, values=["jsonl", "json", "txt"], variable=self.export_format, width=80,
                    fg_color=COLORS["bg_input"], dropdown_fg_color=COLORS["bg_card"]).pack(side="left", padx=5)
        row += 1

        # Section: Advanced Generation Modes
        self._section_header(scroll, "🧪 ADVANCED GENERATION MODES", row); row += 1

        modes_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        modes_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

        CTkCheckBox(modes_frame, text="🗡️ 攻击模式", variable=self.enable_attack,
                    fg_color=COLORS["accent_red"]).grid(row=0, column=0, sticky="w", padx=15, pady=6)
        CTkCheckBox(modes_frame, text="🛡️ 幻觉防御", variable=self.enable_hallucination_defense,
                    fg_color=COLORS["accent_orange"]).grid(row=0, column=1, sticky="w", padx=15, pady=6)
        CTkCheckBox(modes_frame, text="🧠 深度思考", variable=self.enable_deep_thinking,
                    fg_color=COLORS["accent_purple"]).grid(row=1, column=0, sticky="w", padx=15, pady=6)
        CTkCheckBox(modes_frame, text="⚡ 快速模式", variable=self.enable_quick_mode,
                    fg_color=COLORS["accent_green"]).grid(row=1, column=1, sticky="w", padx=15, pady=6)
        row += 1

        # =================== ACTION BUTTONS ===================
        btn_frame = CTkFrame(scroll, fg_color="transparent")
        btn_frame.grid(row=row, column=0, columnspan=3, pady=20)

        self.btn_run = CTkButton(
            btn_frame, text="▶ START FULL GENERATION", command=self.start,
            fg_color=COLORS["accent_green"], text_color="black",
            font=("Consolas", 14, "bold"), width=250, height=50, corner_radius=8
        )
        self.btn_run.pack(side="left", padx=10)

        self.btn_quick = CTkButton(
            btn_frame, text="📊 QUICK SCAN", command=self.quick_scan,
            fg_color=COLORS["accent_blue"], text_color="black",
            font=("Consolas", 14, "bold"), width=180, height=50, corner_radius=8
        )
        self.btn_quick.pack(side="left", padx=10)

        self.btn_stop = CTkButton(
            btn_frame, text="⬛ STOP", command=self.stop, state="disabled",
            fg_color=COLORS["accent_red"], text_color="white",
            font=("Consolas", 14, "bold"), width=120, height=50, corner_radius=8
        )
        self.btn_stop.pack(side="left", padx=10)

        self.btn_force_gpu = CTkButton(
            btn_frame, text="🔥 FORCE GPU", command=self.toggle_force_gpu,
            fg_color=COLORS["accent_orange"], text_color="black",
            font=("Consolas", 12, "bold"), width=140, height=50, corner_radius=8
        )
        self.btn_force_gpu.pack(side="left", padx=10)

        self.btn_unload = CTkButton(
            btn_frame, text="🗑️ UNLOAD", command=self.unload_models,
            fg_color=COLORS["border"],
            font=("Consolas", 12), width=100, height=50, corner_radius=8
        )
        self.btn_unload.pack(side="left", padx=10)

    def _build_engine_tab(self, parent):
        """Inference engine configuration"""
        scroll = CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        CTkLabel(scroll, text="⚙️ INFERENCE ENGINE CONFIGURATION",
                 font=("Consolas", 16, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(anchor="w", padx=10, pady=(5, 15))

        # GPU Layers
        gpu_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        gpu_frame.pack(fill="x", padx=10, pady=5)

        CTkLabel(gpu_frame, text="🎮 GPU OFFLOAD LAYERS",
                 font=("Consolas", 13, "bold"), text_color=COLORS["accent_green"]
                 ).pack(anchor="w", padx=15, pady=(10, 5))

        layers_inner = CTkFrame(gpu_frame, fg_color="transparent")
        layers_inner.pack(fill="x", padx=15, pady=5)

        CTkLabel(layers_inner, text="n_gpu_layers:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).pack(side="left", padx=5)

        self.gpu_layers_slider = CTkSlider(
            layers_inner, from_=-1, to=999, number_of_steps=1000,
            variable=self.n_gpu_layers,
            command=lambda v: self.gpu_layers_label.configure(
                text=f"{'AUTO' if int(v) == -1 else ('CPU ONLY' if int(v) == 0 else str(int(v)))}"
            ),
            fg_color=COLORS["gauge_bg"], progress_color=COLORS["accent_green"]
        )
        self.gpu_layers_slider.pack(side="left", fill="x", expand=True, padx=10)

        self.gpu_layers_label = CTkLabel(layers_inner, text="AUTO", width=80,
                                          font=("Consolas", 13, "bold"),
                                          text_color=COLORS["accent_green"])
        self.gpu_layers_label.pack(side="left", padx=5)

        CTkLabel(gpu_frame,
                 text="  -1=AUTO (all layers to GPU) | 0=CPU only | 1-999=specific layer count",
                 font=("Consolas", 10), text_color=COLORS["text_dim"]
                 ).pack(anchor="w", padx=15, pady=(0, 10))

        # Context & Batch
        ctx_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        ctx_frame.pack(fill="x", padx=10, pady=5)

        CTkLabel(ctx_frame, text="📐 CONTEXT & BATCH SIZE",
                 font=("Consolas", 13, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(anchor="w", padx=15, pady=(10, 5))

        ctx_inner = CTkFrame(ctx_frame, fg_color="transparent")
        ctx_inner.pack(fill="x", padx=15, pady=5)

        CTkLabel(ctx_inner, text="n_ctx:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).pack(side="left", padx=5)
        CTkComboBox(ctx_inner, values=["2048", "4096", "8192", "16384", "32768", "65536", "131072"],
                    variable=ctk.StringVar(value=str(self.n_ctx.get())),
                    command=lambda v: self.n_ctx.set(int(v)),
                    fg_color=COLORS["bg_input"], dropdown_fg_color=COLORS["bg_card"], width=100
                    ).pack(side="left", padx=5)

        CTkLabel(ctx_inner, text="n_batch:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).pack(side="left", padx=15)
        CTkComboBox(ctx_inner, values=["128", "256", "512", "1024", "2048"],
                    variable=ctk.StringVar(value=str(self.n_batch.get())),
                    command=lambda v: self.n_batch.set(int(v)),
                    fg_color=COLORS["bg_input"], dropdown_fg_color=COLORS["bg_card"], width=100
                    ).pack(side="left", padx=5)

        CTkLabel(ctx_inner, text="n_threads:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).pack(side="left", padx=15)
        CTkEntry(ctx_inner, textvariable=self.n_threads, width=60,
                 fg_color=COLORS["bg_input"]).pack(side="left", padx=5)
        CTkLabel(ctx_inner, text="(0=auto)", font=("Consolas", 10),
                 text_color=COLORS["text_dim"]).pack(side="left")

        ctx_inner.pack(fill="x", padx=15, pady=(5, 10))

        # Advanced Engine Options
        adv_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        adv_frame.pack(fill="x", padx=10, pady=5)

        CTkLabel(adv_frame, text="🔧 ADVANCED ENGINE OPTIONS",
                 font=("Consolas", 13, "bold"), text_color=COLORS["accent_orange"]
                 ).pack(anchor="w", padx=15, pady=(10, 5))

        adv_inner = CTkFrame(adv_frame, fg_color="transparent")
        adv_inner.pack(fill="x", padx=15, pady=(5, 10))

        CTkCheckBox(adv_inner, text="Flash Attention", variable=self.flash_attention,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15)
        CTkCheckBox(adv_inner, text="Memory Map (mmap)", variable=self.use_mmap,
                    fg_color=COLORS["accent_blue"]).pack(side="left", padx=15)
        CTkCheckBox(adv_inner, text="llama.cpp Server ★", variable=self.is_llama_server,
                    fg_color=COLORS["accent_orange"]).pack(side="left", padx=15)

        CTkLabel(adv_inner, text="Seed:", font=("Consolas", 12),
                 text_color=COLORS["text_secondary"]).pack(side="left", padx=15)
        CTkEntry(adv_inner, textvariable=self.seed, width=80,
                 fg_color=COLORS["bg_input"]).pack(side="left", padx=5)
        CTkLabel(adv_inner, text="(-1=random)", font=("Consolas", 10),
                 text_color=COLORS["text_dim"]).pack(side="left")

        # Recommended layers
        rec_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=5)

        CTkLabel(rec_frame, text="💡 GPU LAYER RECOMMENDATION",
                 font=("Consolas", 13, "bold"), text_color=COLORS["accent_yellow"]
                 ).pack(anchor="w", padx=15, pady=(10, 5))

        self.rec_label = CTkLabel(
            rec_frame, text="Click 'Analyze' to get GPU layer recommendation based on available VRAM",
            font=("Consolas", 11), text_color=COLORS["text_secondary"]
        )
        self.rec_label.pack(anchor="w", padx=15, pady=5)

        CTkButton(rec_frame, text="🔍 Analyze VRAM", command=self._analyze_vram,
                  fg_color=COLORS["accent_blue"], text_color="black", width=160
                  ).pack(anchor="w", padx=15, pady=(0, 10))

        # Save/Load config
        cfg_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        cfg_frame.pack(fill="x", padx=10, pady=5)

        CTkLabel(cfg_frame, text="💾 CONFIGURATION",
                 font=("Consolas", 13, "bold"), text_color=COLORS["accent_purple"]
                 ).pack(anchor="w", padx=15, pady=(10, 5))

        cfg_btns = CTkFrame(cfg_frame, fg_color="transparent")
        cfg_btns.pack(fill="x", padx=15, pady=(5, 10))

        CTkButton(cfg_btns, text="Save Config", command=self.save_config,
                  fg_color=COLORS["accent_green"], text_color="black", width=130
                  ).pack(side="left", padx=5)
        CTkButton(cfg_btns, text="Load Config", command=self.load_config,
                  fg_color=COLORS["accent_blue"], text_color="black", width=130
                  ).pack(side="left", padx=5)

    def _build_sampling_tab(self, parent):
        """Sampling parameters"""
        scroll = CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        CTkLabel(scroll, text="🎯 SAMPLING PARAMETERS",
                 font=("Consolas", 16, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(anchor="w", padx=10, pady=(5, 15))

        params = [
            ("Temperature", self.temperature, 0.0, 2.0, 200, "Controls randomness. Lower = more deterministic."),
            ("Top P", self.top_p, 0.0, 1.0, 100, "Nucleus sampling. 0.95 = consider top 95% probability mass."),
            ("Min P", self.min_p, 0.0, 0.5, 50, "Minimum probability threshold for token selection."),
            ("Repeat Penalty", self.repeat_penalty, 1.0, 2.0, 100, "Penalize repeated tokens. 1.0 = no penalty."),
            ("Presence Penalty", self.presence_penalty, -2.0, 2.0, 400, "Penalize tokens already present in context."),
            ("Frequency Penalty", self.frequency_penalty, -2.0, 2.0, 400, "Penalize tokens based on frequency."),
        ]

        for name, var, min_v, max_v, steps, desc in params:
            frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
            frame.pack(fill="x", padx=10, pady=4)

            header = CTkFrame(frame, fg_color="transparent")
            header.pack(fill="x", padx=15, pady=(8, 2))

            CTkLabel(header, text=name, font=("Consolas", 13, "bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")
            val_label = CTkLabel(header, text=f"{var.get():.2f}", font=("Consolas", 13, "bold"),
                                  text_color=COLORS["accent_blue"], width=70)
            val_label.pack(side="right")

            slider = CTkSlider(
                frame, from_=min_v, to=max_v, number_of_steps=steps, variable=var,
                command=lambda v, lbl=val_label: lbl.configure(text=f"{float(v):.2f}"),
                fg_color=COLORS["gauge_bg"], progress_color=COLORS["accent_blue"]
            )
            slider.pack(fill="x", padx=15, pady=2)

            CTkLabel(frame, text=desc, font=("Consolas", 10),
                     text_color=COLORS["text_dim"]).pack(anchor="w", padx=15, pady=(0, 8))

        # Top K (integer)
        topk_frame = CTkFrame(scroll, fg_color=COLORS["bg_card"], corner_radius=8)
        topk_frame.pack(fill="x", padx=10, pady=4)

        topk_header = CTkFrame(topk_frame, fg_color="transparent")
        topk_header.pack(fill="x", padx=15, pady=(8, 2))

        CTkLabel(topk_header, text="Top K", font=("Consolas", 13, "bold"),
                 text_color=COLORS["text_primary"]).pack(side="left")
        self.topk_label = CTkLabel(topk_header, text=str(self.top_k.get()),
                                    font=("Consolas", 13, "bold"),
                                    text_color=COLORS["accent_blue"], width=70)
        self.topk_label.pack(side="right")

        CTkSlider(
            topk_frame, from_=1, to=200, number_of_steps=199, variable=self.top_k,
            command=lambda v: self.topk_label.configure(text=str(int(v))),
            fg_color=COLORS["gauge_bg"], progress_color=COLORS["accent_blue"]
        ).pack(fill="x", padx=15, pady=2)

        CTkLabel(topk_frame, text="Number of top tokens to consider. 40 = only top 40 most likely tokens.",
                 font=("Consolas", 10), text_color=COLORS["text_dim"]).pack(anchor="w", padx=15, pady=(0, 8))

    def _build_prompt_tab(self, parent):
        """Prompt editor"""
        frame = CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        CTkLabel(frame, text="📝 SYSTEM PROMPT EDITOR",
                 font=("Consolas", 16, "bold"), text_color=COLORS["accent_blue"]
                 ).pack(anchor="w", padx=10, pady=(5, 10))

        self.prompt_textbox = CTkTextbox(frame, font=("Consolas", 12),
                                          fg_color=COLORS["bg_input"])
        self.prompt_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.prompt_textbox.insert("1.0", DEFAULT_SYSTEM_PROMPT)

        btn_frame = CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        CTkButton(btn_frame, text="✅ Apply", command=self.apply_prompt,
                  fg_color=COLORS["accent_green"], text_color="black", width=120
                  ).pack(side="left", padx=10)
        CTkButton(btn_frame, text="🔄 Reset Default", command=self.reset_prompt,
                  fg_color=COLORS["accent_orange"], text_color="black", width=140
                  ).pack(side="left", padx=10)
        CTkButton(btn_frame, text="📂 Load from File", command=self.load_prompt_file,
                  fg_color=COLORS["border"], width=140
                  ).pack(side="left", padx=10)
        CTkButton(btn_frame, text="💾 Save to File", command=self.save_prompt_file,
                  fg_color=COLORS["border"], width=140
                  ).pack(side="left", padx=10)

    def _build_preview_tab(self, parent):
        """Real-time generation preview"""
        frame = CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Metrics panel at top
        self.metrics_panel = MetricsPanel(frame)
        self.metrics_panel.pack(fill="x", padx=5, pady=(0, 5))

        self.preview_box = CTkTextbox(frame, font=("Consolas", 12),
                                       fg_color=COLORS["bg_input"], wrap="word")
        self.preview_box.pack(fill="both", expand=True, padx=5, pady=5)

        export_frame = CTkFrame(frame, fg_color="transparent")
        export_frame.pack(fill="x", padx=5, pady=5)

        CTkButton(export_frame, text="Export TXT", command=lambda: self.export_preview("txt"),
                  fg_color=COLORS["border"], width=110).pack(side="left", padx=5)
        CTkButton(export_frame, text="Export JSON", command=lambda: self.export_preview("json"),
                  fg_color=COLORS["border"], width=110).pack(side="left", padx=5)
        CTkButton(export_frame, text="Clear Preview", command=lambda: self.preview_box.delete("1.0", "end"),
                  fg_color=COLORS["accent_red"], text_color="white", width=110).pack(side="right", padx=5)

    def _build_history_tab(self, parent):
        """Generation history"""
        self.history_box = CTkTextbox(parent, font=("Consolas", 11),
                                       fg_color=COLORS["bg_input"])
        self.history_box.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_log_tab(self, parent):
        """Run log"""
        self.log_box = CTkTextbox(parent, font=("Consolas", 11),
                                   fg_color=COLORS["bg_input"])
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_chart_tab(self, parent):
        """Discipline statistics chart"""
        chart_frame = CTkFrame(parent, fg_color="transparent")
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.figure = Figure(figsize=(14, 6), dpi=100, facecolor=COLORS["bg_dark"])
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        btn_frame = CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", pady=5)
        CTkButton(btn_frame, text="🔄 Refresh", command=self.update_chart,
                  fg_color=COLORS["accent_blue"], text_color="black", width=100).pack(side="right", padx=10)
        CTkButton(btn_frame, text="🗑️ Clear", command=self.clear_stats,
                  fg_color=COLORS["accent_red"], text_color="white", width=100).pack(side="right", padx=5)

    def _build_status_bar(self):
        """Bottom status bar"""
        bar = CTkFrame(self, fg_color=COLORS["bg_panel"], height=36, corner_radius=0)
        bar.pack(fill="x", side="bottom")

        self.progress = CTkProgressBar(bar, fg_color=COLORS["gauge_bg"],
                                        progress_color=COLORS["accent_blue"], height=6)
        self.progress.pack(fill="x", padx=10, pady=(4, 0))
        self.progress.set(0)

        status_inner = CTkFrame(bar, fg_color="transparent")
        status_inner.pack(fill="x", padx=10, pady=2)

        self.status_lbl = CTkLabel(status_inner, text="READY",
                                    font=("Consolas", 11), text_color=COLORS["accent_green"])
        self.status_lbl.pack(side="left")

        self.gpu_status_lbl = CTkLabel(status_inner, text="GPU: Detecting...",
                                        font=("Consolas", 11), text_color=COLORS["text_dim"])
        self.gpu_status_lbl.pack(side="right")

        self.force_gpu_lbl = CTkLabel(status_inner, text="",
                                       font=("Consolas", 11, "bold"), text_color=COLORS["accent_orange"])
        self.force_gpu_lbl.pack(side="right", padx=15)

    # ======================== HELPER METHODS ========================

    def _section_header(self, parent, text, row):
        CTkLabel(parent, text=text, font=("Consolas", 14, "bold"),
                 text_color=COLORS["accent_blue"]).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=10, pady=(15, 5)
        )

    def _on_gpu_update(self, gpus, sys_info):
        """Callback from GPU monitor - update dashboard (thread-safe)"""
        try:
            self.after(0, self.gpu_dashboard.update_display, gpus, sys_info)
            gpu = gpus[0] if gpus else None
            if gpu and gpu.is_available:
                self.after(0, self.gpu_status_lbl.configure,
                          {"text": f"GPU: {gpu.name} | VRAM: {gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f}GB | {gpu.gpu_utilization:.0f}%"})
            else:
                self.after(0, self.gpu_status_lbl.configure,
                          {"text": "GPU: NONE (CPU mode)", "text_color": COLORS["accent_red"]})
        except Exception:
            pass

    def _log_startup_info(self):
        self.log(f"Night Cruise v{VERSION} — Industrial Training Platform")
        self.log(f"GPU Backend: {self.gpu_monitor.backend.upper()}")
        gpu = self.gpu_monitor.primary_gpu
        if gpu and gpu.is_available:
            self.log(f"GPU: {gpu.name} | VRAM: {gpu.memory_total_gb:.1f} GB")
            self.log(f"GPU Status: ✅ ONLINE — GPU inference available")
        else:
            self.log("⚠️ WARNING: No GPU detected — CPU inference mode")
            self.log("Performance will be severely degraded. Consider installing CUDA drivers.")
        sys_info = self.gpu_monitor.system_info
        self.log(f"System: {sys_info.os_name} | CPU: {sys_info.cpu_count} cores | RAM: {sys_info.ram_total_gb:.1f} GB")
        self.log("System ready.")

    def _build_inference_config(self) -> InferenceConfig:
        return InferenceConfig(
            provider=self.provider.get(),
            base_url=self.base_url.get(),
            api_key=self.api_key.get(),
            gguf_path=self.gguf_path.get(),
            hf_model_id=self.hf_model_id.get(),
            model=self.model.get(),
            n_ctx=self.n_ctx.get(),
            n_gpu_layers=self.n_gpu_layers.get(),
            n_batch=self.n_batch.get(),
            n_threads=self.n_threads.get(),
            flash_attention=self.flash_attention.get(),
            use_mmap=self.use_mmap.get(),
            seed=self.seed.get(),
            is_llama_server=self.is_llama_server.get(),
            verbose=False,
        )

    def _build_gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            system_prompt=self.current_system_prompt,
            max_tokens=self.num_predict.get(),
            temperature=self.temperature.get(),
            top_p=self.top_p.get(),
            top_k=self.top_k.get(),
            min_p=self.min_p.get(),
            repeat_penalty=self.repeat_penalty.get(),
            presence_penalty=self.presence_penalty.get(),
            frequency_penalty=self.frequency_penalty.get(),
        )

    def _analyze_vram(self):
        gpu = self.gpu_monitor.primary_gpu
        if not gpu or not gpu.is_available:
            self.rec_label.configure(
                text="❌ No GPU available. Use CPU mode (n_gpu_layers=0).",
                text_color=COLORS["accent_red"]
            )
            return

        free_gb = gpu.memory_free_gb
        rec = self.gpu_monitor.get_recommended_gpu_layers()
        self.rec_label.configure(
            text=f"✅ {gpu.name} | Free VRAM: {free_gb:.1f} GB | "
                 f"Recommended layers: ~{rec} (for ~4GB model) | "
                 f"Set to -1 for auto-detect",
            text_color=COLORS["accent_green"]
        )

    # ======================== UI INTERACTIONS ========================

    def pick_paper_dir(self):
        d = filedialog.askdirectory(title="选择论文文件夹")
        if d:
            self.paper_dir.set(d)

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.out_dir.set(d)

    def pick_gguf(self):
        f = filedialog.askopenfilename(title="选择 GGUF 模型文件", filetypes=[("GGUF", "*.gguf"), ("All", "*.*")])
        if f:
            self.gguf_path.set(f)

    def pick_validator_gguf(self):
        f = filedialog.askopenfilename(title="选择验证器 GGUF", filetypes=[("GGUF", "*.gguf"), ("All", "*.*")])
        if f:
            self.validator_gguf.set(f)

    def update_provider(self, *args):
        prov = self.provider.get()
        states = {
            "ollama":           (True,  False, False, False),
            "openai_compatible":(True,  True,  False, False),
            "local_gguf":       (False, False, True,  False),
            "huggingface":      (False, False, False, True),
        }
        url_on, key_on, gguf_on, hf_on = states.get(prov, (False, False, False, False))

        self.base_url_entry.configure(state="normal" if url_on else "disabled")
        self.api_key_entry.configure(state="normal" if key_on else "disabled")
        self.gguf_entry.configure(state="normal" if gguf_on else "disabled")
        self.gguf_button.configure(state="normal" if gguf_on else "disabled")
        self.hf_entry.configure(state="normal" if hf_on else "disabled")
        self.refresh_models()

    def on_dual_toggle(self, *args):
        self.second_model_combo.configure(state="readonly" if self.enable_dual.get() else "disabled")

    def on_validator_toggle(self, *args):
        st = "normal" if self.enable_validator.get() else "disabled"
        self.validator_entry.configure(state=st)

    def refresh_models(self):
        models = get_models(self.provider.get(), self.base_url.get(), self.api_key.get())
        self.model_combo.configure(values=models)
        self.second_model_combo.configure(values=models)
        if models and not models[0].startswith("Error") and not models[0].startswith("("):
            if self.model.get() not in models:
                self.model.set(models[0])

    def test_api(self):
        models = get_models(self.provider.get(), self.base_url.get(), self.api_key.get())
        if models[0].startswith("Error") or models[0].startswith("HTTP"):
            messagebox.showerror("Connection Failed", models[0])
        else:
            messagebox.showinfo("Connected", f"Found {len(models)} model(s)")

    def toggle_force_gpu(self):
        current = self.force_gpu.get()
        self.force_gpu.set(not current)
        if not current:
            if not self.gpu_monitor.has_gpu:
                messagebox.showwarning(
                    "⚠️ No GPU Available",
                    "Force GPU mode enabled but no GPU was detected!\n"
                    "This may cause errors. Ensure CUDA/ROCm drivers are installed."
                )
            self.btn_force_gpu.configure(fg_color=COLORS["accent_red"], text="🔥 GPU FORCED")
            self.force_gpu_lbl.configure(text="⚡ FORCE GPU: ON")
            self.log("🔥 FORCE GPU MODE: ENABLED — all layers will be offloaded to GPU")
        else:
            self.btn_force_gpu.configure(fg_color=COLORS["accent_orange"], text="🔥 FORCE GPU")
            self.force_gpu_lbl.configure(text="")
            self.log("Force GPU mode disabled")

    def unload_models(self):
        if self._inference_engine:
            self._inference_engine.unload_models()
            self._inference_engine = None
        gc.collect()
        self.log("All models unloaded. VRAM freed.")

    def log(self, msg: str):
        ts = now_time()
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")

    def preview_append(self, text: str):
        self.preview_box.insert("end", text)
        self.preview_box.see("end")

    def apply_prompt(self):
        self.current_system_prompt = self.prompt_textbox.get("1.0", "end").strip()
        self.log("✅ System prompt applied")

    def reset_prompt(self):
        self.prompt_textbox.delete("1.0", "end")
        self.prompt_textbox.insert("1.0", DEFAULT_SYSTEM_PROMPT)
        self.current_system_prompt = DEFAULT_SYSTEM_PROMPT
        self.log("🔄 System prompt reset to default")

    def load_prompt_file(self):
        f = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if f:
            content = Path(f).read_text(encoding="utf-8")
            self.prompt_textbox.delete("1.0", "end")
            self.prompt_textbox.insert("1.0", content)
            self.current_system_prompt = content
            self.log(f"Prompt loaded from {f}")

    def save_prompt_file(self):
        f = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if f:
            Path(f).write_text(self.current_system_prompt, encoding="utf-8")
            self.log(f"Prompt saved to {f}")

    def add_to_history(self, input_name, output_name, success):
        status = "✅" if success else "❌"
        entry = f"[{now_ts()}] {status} {input_name} → {output_name}\n"
        self.history_box.insert("end", entry)
        self.history_box.see("end")
        self.history.append({"time": now_ts(), "input": input_name, "output": output_name, "status": status})

    def export_preview(self, fmt):
        content = self.preview_box.get("1.0", "end-1c").strip()
        if not content:
            messagebox.showwarning("Empty", "Preview is empty")
            return
        f = filedialog.asksaveasfilename(defaultextension=f".{fmt}",
                                          filetypes=[(f"{fmt.upper()}", f"*.{fmt}")])
        if f:
            try:
                if fmt == "json":
                    json.dump({"preview": content}, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                else:
                    open(f, "w", encoding="utf-8").write(content)
                self.log(f"Preview exported: {f}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def save_config(self):
        f = filedialog.asksaveasfilename(defaultextension=".json",
                                          filetypes=[("JSON", "*.json")])
        if not f:
            return
        cfg = {
            "provider": self.provider.get(),
            "base_url": self.base_url.get(),
            "model": self.model.get(),
            "gguf_path": self.gguf_path.get(),
            "hf_model_id": self.hf_model_id.get(),
            "n_gpu_layers": self.n_gpu_layers.get(),
            "n_ctx": self.n_ctx.get(),
            "n_batch": self.n_batch.get(),
            "n_threads": self.n_threads.get(),
            "temperature": self.temperature.get(),
            "top_p": self.top_p.get(),
            "top_k": self.top_k.get(),
            "min_p": self.min_p.get(),
            "repeat_penalty": self.repeat_penalty.get(),
            "presence_penalty": self.presence_penalty.get(),
            "frequency_penalty": self.frequency_penalty.get(),
            "max_tokens": self.num_predict.get(),
            "max_chars": self.max_chars.get(),
            "export_format": self.export_format.get(),
            "flash_attention": self.flash_attention.get(),
            "use_mmap": self.use_mmap.get(),
            "seed": self.seed.get(),
            "is_llama_server": self.is_llama_server.get(),
        }
        Path(f).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self.log(f"Config saved: {f}")

    def load_config(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not f:
            return
        try:
            cfg = json.loads(Path(f).read_text(encoding="utf-8"))
            var_map = {
                "provider": self.provider, "base_url": self.base_url,
                "model": self.model, "gguf_path": self.gguf_path,
                "hf_model_id": self.hf_model_id,
                "export_format": self.export_format,
            }
            int_map = {
                "n_gpu_layers": self.n_gpu_layers, "n_ctx": self.n_ctx,
                "n_batch": self.n_batch, "n_threads": self.n_threads,
                "top_k": self.top_k, "max_tokens": self.num_predict,
                "max_chars": self.max_chars, "seed": self.seed,
            }
            float_map = {
                "temperature": self.temperature, "top_p": self.top_p,
                "min_p": self.min_p, "repeat_penalty": self.repeat_penalty,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
            }
            bool_map = {
                "flash_attention": self.flash_attention, "use_mmap": self.use_mmap,
                "is_llama_server": self.is_llama_server,
            }
            for k, v in var_map.items():
                if k in cfg: v.set(str(cfg[k]))
            for k, v in int_map.items():
                if k in cfg: v.set(int(cfg[k]))
            for k, v in float_map.items():
                if k in cfg: v.set(float(cfg[k]))
            for k, v in bool_map.items():
                if k in cfg: v.set(bool(cfg[k]))

            self.update_provider()
            self.log(f"Config loaded: {f}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def clear_stats(self):
        self.major_counter.clear()
        self.sub_counter.clear()
        self.update_chart()
        self.log("Statistics cleared")

    def update_chart(self):
        self.figure.clear()
        self.figure.set_facecolor(COLORS["bg_dark"])

        if not self.major_counter:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(COLORS["bg_dark"])
            ax.text(0.5, 0.5, "No data yet\nRun processing or quick scan first",
                    ha="center", va="center", fontsize=16, color=COLORS["text_dim"])
            ax.axis('off')
            self.canvas.draw()
            return

        ax1 = self.figure.add_subplot(121)
        ax1.set_facecolor(COLORS["bg_dark"])
        colors_pie = [COLORS["accent_blue"], COLORS["accent_green"], COLORS["accent_orange"],
                      COLORS["accent_purple"], COLORS["accent_red"], COLORS["accent_yellow"]]
        ax1.pie(self.major_counter.values(), labels=self.major_counter.keys(),
                autopct='%1.1f%%', startangle=90, colors=colors_pie[:len(self.major_counter)],
                textprops={'color': COLORS["text_primary"], 'fontsize': 10})
        ax1.set_title("Major Disciplines", color=COLORS["text_primary"], fontsize=13)

        ax2 = self.figure.add_subplot(122)
        ax2.set_facecolor(COLORS["bg_dark"])
        top_subs = self.sub_counter.most_common(15)
        if top_subs:
            subs, counts = zip(*top_subs)
            bars = ax2.barh(range(len(subs)), counts, color=COLORS["accent_blue"])
            ax2.set_yticks(range(len(subs)))
            ax2.set_yticklabels(subs, color=COLORS["text_secondary"], fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel("Papers", color=COLORS["text_secondary"])
            ax2.set_title("Top Sub-Disciplines", color=COLORS["text_primary"], fontsize=13)
            ax2.tick_params(colors=COLORS["text_dim"])
            ax2.spines['bottom'].set_color(COLORS["border"])
            ax2.spines['left'].set_color(COLORS["border"])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        self.figure.tight_layout()
        self.canvas.draw()

    # ======================== TASK EXECUTION ========================

    def start(self):
        if not self.paper_dir.get():
            messagebox.showerror("Error", "请选择论文文件夹")
            return
        if self.provider.get() == "local_gguf" and not self.gguf_path.get():
            messagebox.showerror("Error", "请选择 GGUF 模型文件")
            return
        if self.provider.get() == "huggingface" and not self.hf_model_id.get():
            messagebox.showerror("Error", "请输入 HuggingFace Model ID")
            return

        # Check GPU status
        if self.force_gpu.get() and not self.gpu_monitor.has_gpu:
            if not messagebox.askyesno(
                "⚠️ GPU Not Found",
                "Force GPU is ON but no GPU was detected.\n"
                "This will likely cause an error.\n\nContinue anyway?"
            ):
                return

        self.btn_run.configure(state="disabled")
        self.btn_quick.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._stop_flag = False
        self.status_lbl.configure(text="RUNNING...", text_color=COLORS["accent_orange"])
        threading.Thread(target=self._run_job, daemon=True).start()
        self.log("▶ Starting full generation job (v7.1 Fresh Session mode)...")

    def stop(self):
        self._stop_flag = True
        self.log("⏹ Stopping task...")
        self.status_lbl.configure(text="STOPPING...", text_color=COLORS["accent_red"])

    def quick_scan(self):
        if not self.paper_dir.get():
            messagebox.showerror("Error", "请选择论文文件夹")
            return
        paper_dir = Path(self.paper_dir.get())
        pattern = "**/*.*" if self.recursive.get() else "*.*"
        files = [p for p in paper_dir.rglob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        if not files:
            messagebox.showinfo("Done", "No supported files found")
            return

        self.major_counter.clear()
        self.sub_counter.clear()
        for path in files:
            text = extract_text(path)
            if "[ERROR]" in text:
                continue
            abstract = heuristic_abstract(text, 10000)
            major, sub = classify_paper(abstract)
            self.major_counter[major] += 1
            self.sub_counter[f"{major} - {sub}"] += 1

        self.update_chart()
        self.log(f"Quick scan complete: {len(files)} files")
        messagebox.showinfo("Done", f"Scanned {len(files)} files")

    def _run_job(self):
        paper_dir = Path(self.paper_dir.get())
        out_dir = Path(self.out_dir.get())
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = "**/*.*" if self.recursive.get() else "*.*"
        files = [p for p in paper_dir.rglob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        total = len(files)
        if total == 0:
            self.after(0, lambda: messagebox.showinfo("Done", "No files found"))
            self.after(0, self._reset_buttons)
            return

        # Build engine
        inf_config = self._build_inference_config()
        engine = InferenceEngine(inf_config)
        self._inference_engine = engine

        gen_config = self._build_gen_config()
        force_gpu = self.force_gpu.get()

        # ★ v7.1: 显示后端模式信息
        if inf_config.is_llama_server and inf_config.provider == "openai_compatible":
            self.after(0, self.log,
                "🔧 llama.cpp Server 模式已启用 — 将自动清除 slot KV Cache")
        elif inf_config.provider == "ollama":
            self.after(0, self.log,
                "🔧 Ollama 模式 — 将使用 keep_alive=0 刷新上下文")

        # CPU inference warning
        if not self.gpu_monitor.has_gpu and not force_gpu:
            self.after(0, self.log, "⚠️ ALERT: Running on CPU — inference will be slow!")

        def generate_output(model_name, system, user_prompt, np_override=None, temp_override=None):
            full = ""
            def cb(token):
                nonlocal full
                full += token
                self.after(0, self.preview_append, token)

            gc = GenerationConfig(
                system_prompt=system,
                max_tokens=np_override or gen_config.max_tokens,
                temperature=temp_override if temp_override is not None else gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                min_p=gen_config.min_p,
                repeat_penalty=gen_config.repeat_penalty,
                presence_penalty=gen_config.presence_penalty,
                frequency_penalty=gen_config.frequency_penalty,
            )

            try:
                # ★ v7.1: 每次都创建全新的 InferenceEngine 实例
                # 确保不会复用上一次的连接或 KV Cache
                eng_cfg = InferenceConfig(**{**inf_config.__dict__, "model": model_name})
                eng = InferenceEngine(eng_cfg)
                text, metrics = eng.generate_stream(user_prompt, gc, cb, force_gpu=force_gpu)
                self.after(0, self.metrics_panel.update_metrics, metrics)
                self.after(0, self.log,
                    f"🔄 Session #{metrics.session_id} | "
                    f"{metrics.tokens_per_second:.1f} tok/s | "
                    f"{metrics.tokens_generated} tokens"
                )
                return text
            except Exception as e:
                self.after(0, self.log, f"❌ {model_name} error: {e}")
                return ""

        for idx, path in enumerate(files):
            if self._stop_flag:
                break

            # ★ v7.1: 每个文件开始前，重置引擎上下文 (刷掉上次的 KV Cache)
            engine.reset_context()
            self.after(0, self.log,
                f"🆕 File {idx+1}/{total}: 已开启全新 API 会话窗口")

            self.after(0, lambda: self.preview_box.delete("1.0", "end"))
            self.after(0, lambda p=path: self.preview_box.insert("end", f"📄 Processing: {p.name}\n\n"))

            text = extract_text(path)
            if "[ERROR]" in text:
                self.after(0, self.log, f"❌ Extraction failed: {path.name}")
                continue

            abstract = heuristic_abstract(text, self.max_chars.get())
            user_prompt = f"Provided text (likely abstract or excerpt):\n\n{abstract}"

            effective_np = self.num_predict.get()
            if self.enable_quick_mode.get():
                effective_np = max(512, effective_np // 2)
            if self.enable_deep_thinking.get():
                effective_np = min(8192, effective_np * 2)

            primary_output = generate_output(self.model.get(), self.current_system_prompt,
                                              user_prompt, effective_np)

            # Dual model
            use_second = self.enable_dual.get() and self.second_model.get()
            if use_second:
                need_review = self.enable_attack.get() or self.enable_hallucination_defense.get()
                if need_review:
                    review_system = "You are a strict academic critic.\n"
                    if self.enable_attack.get():
                        review_system += "Be aggressive: question every claim, refute weak evidence.\n"
                    if self.enable_hallucination_defense.get():
                        review_system += "Focus on hallucination detection and correction.\n"
                    review_user = f"Review:\n\n{primary_output}"
                    self.after(0, self.preview_append, "\n\n[S8] SECOND MODEL REVIEW\n")
                    review = generate_output(self.second_model.get(), review_system, review_user, 2048)
                    primary_output += "\n\n[S8] SECOND MODEL REVIEW\n" + review
                else:
                    self.after(0, self.preview_append, "\n\n[ALTERNATIVE]\n")
                    alt = generate_output(self.second_model.get(), self.current_system_prompt,
                                           user_prompt, effective_np)
                    primary_output += "\n\n[ALTERNATIVE]\n" + alt

            # Quality guard
            if self.enable_quality_guard.get():
                ok, reasons = quality_guard(primary_output)
                if not ok:
                    self.after(0, self.log, f"⚠️ Quality guard failed {path.name}: {', '.join(reasons)}")
                    continue
            else:
                self.after(0, self.log, f"Quality guard OFF — saving {path.name}")

            # Validator
            validation = {"score": 10.0, "issues": [], "confidence": "high"}
            if self.enable_validator.get() and Path(self.validator_gguf.get()).exists():
                try:
                    v_prompt = VALIDATOR_PROMPT.format(generated_output=primary_output)
                    v_cfg = InferenceConfig(
                        provider="local_gguf", gguf_path=self.validator_gguf.get(),
                        n_ctx=4096, n_gpu_layers=self.n_gpu_layers.get(),
                    )
                    # ★ v7.1: Validator 也用全新引擎实例
                    v_engine = InferenceEngine(v_cfg)
                    v_gen = GenerationConfig(system_prompt="", max_tokens=512, temperature=0.0)
                    v_out, v_metrics = v_engine.generate_stream(
                        v_prompt, v_gen,
                        lambda t: self.after(0, self.preview_append, t),
                        is_validator=True
                    )
                    validation = parse_validator_output(v_out)
                    self.after(0, self.log,
                        f"Validation: {validation['score']} ({validation['confidence']}) "
                        f"| Session #{v_metrics.session_id}")
                except Exception as e:
                    self.after(0, self.log, f"Validator error: {e}")

            # Save output
            ts = now_ts()
            prefix = self.output_prefix.get()
            stem = path.stem
            low = "_low_conf" if validation["score"] < LOW_CONF_THRESHOLD else ""
            fmt = self.export_format.get()

            major, sub = classify_paper(abstract)
            self.major_counter[major] += 1
            self.sub_counter[f"{major} - {sub}"] += 1

            if fmt == "jsonl":
                out_path = out_dir / f"{prefix}_{ts}_{stem}{low}.jsonl"
                content = {"text": primary_output, "validation": validation, "discipline": {"major": major, "sub": sub}}
                out_path.write_text(json.dumps(content, ensure_ascii=False) + "\n", encoding="utf-8")
            elif fmt == "json":
                out_path = out_dir / f"{prefix}_{ts}_{stem}{low}.json"
                content = {"text": primary_output, "validation": validation, "discipline": {"major": major, "sub": sub}}
                out_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                out_path = out_dir / f"{prefix}_{ts}_{stem}{low}.txt"
                out_path.write_text(primary_output + "\n\n--- Validation ---\n" + json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

            self.after(0, self.add_to_history, path.name, out_path.name, True)
            self.after(0, self.log, f"✅ Saved: {out_path.name} | Conf: {validation['confidence']}")
            self.after(0, self.update_chart)
            self.after(0, self.progress.set, (idx + 1) / total)

        # ★ v7.1: 任务完成后最终清理
        engine.reset_context()
        self.after(0, self._reset_buttons)
        self.after(0, self.log, "✅ All tasks complete (all sessions closed)")

    def _reset_buttons(self):
        self.btn_run.configure(state="normal")
        self.btn_quick.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.progress.set(0)
        self.status_lbl.configure(text="READY", text_color=COLORS["accent_green"])

    def auto_backup_preview(self):
        content = self.preview_box.get("1.0", "end-1c").strip()
        if content:
            Path("last_preview_backup.txt").write_text(content, encoding="utf-8")

    def on_closing(self):
        self.auto_backup_preview()
        self.gpu_monitor.stop()
        if self._inference_engine:
            self._inference_engine.unload_models()
        if messagebox.askokcancel("Exit", "Exit Night Cruise?"):
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
