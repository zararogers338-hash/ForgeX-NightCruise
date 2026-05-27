# Monitoring Window / 观测窗说明

NightCruise includes a real-time observation window called **GPU Control Center**.

NightCruise 内置一个实时观测窗，叫 **GPU Control Center**。

## What it observes / 它观察什么

- GPU Load / GPU 负载
- VRAM usage / 显存占用
- GPU temperature / GPU 温度
- Power draw / 功耗
- CPU usage / CPU 占用
- RAM usage / 内存占用
- GPU clock and memory clock / GPU 核心频率与显存频率
- Fan speed / 风扇转速
- Driver information / 驱动信息
- CPU-only fallback warning / CPU 模式警告

## Why it matters / 为什么重要

During long inference or dataset generation jobs, the UI may look calm while the hardware is under heavy load. The observation window helps you verify whether the GPU is actually doing work, whether the job has fallen back to CPU, and whether temperature or memory usage is becoming unsafe.

在长时间推理或数据集生成任务中，界面看起来可能很平静，但硬件已经高负载运行。观测窗可以帮助你确认 GPU 是否真的在工作、任务是否掉回 CPU、温度或显存是否正在接近危险区。

## Safety note / 安全提醒

This monitor is informational. It helps you notice problems, but it cannot physically protect your hardware. If temperature, fan, power, or VRAM readings look abnormal, stop the job and inspect your cooling, drivers, and power settings.

该观测窗主要提供信息。它能帮助你发现问题，但不能物理保护硬件。如果温度、风扇、功耗或显存读数异常，请停止任务，检查散热、驱动和电源设置。
