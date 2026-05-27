# Architecture / 架构说明

ForgeX NightCruise is organized around four layers:

1. **UI Layer (`main.py`)** — CustomTkinter desktop interface, configuration panels, preview window, charts, history, and the GPU Control Center.
2. **Inference Layer (`core/inference.py`)** — Ollama, OpenAI-compatible API, llama.cpp server, local GGUF, and HuggingFace backends.
3. **Document Layer (`core/text_processing.py`)** — file extraction, abstract heuristics, discipline classification, quality guard, validator parsing.
4. **Monitoring Layer (`core/gpu_monitor.py`)** — CPU/RAM/GPU polling, VRAM usage, utilization, temperature, power, clocks, fan speed, and backend detection.

ForgeX NightCruise 的结构可以理解成四层：

1. **界面层 (`main.py`)**：CustomTkinter 桌面界面、配置区、预览区、图表、历史记录和 GPU Control Center。
2. **推理层 (`core/inference.py`)**：对接 Ollama、OpenAI-compatible API、llama.cpp server、本地 GGUF、HuggingFace。
3. **文档层 (`core/text_processing.py`)**：文件提取、摘要截取、学科分类、质量检查、验证器输出解析。
4. **观测层 (`core/gpu_monitor.py`)**：CPU/RAM/GPU 轮询、显存、负载、温度、功耗、频率、风扇、后端检测。
