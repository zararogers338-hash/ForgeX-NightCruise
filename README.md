# 🌙 ForgeX NightCruise v7.1 Industrial

**Synthetic Dataset Factory for Local AI Training**  
**本地 AI 训练用合成数据集工厂**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![UI](https://img.shields.io/badge/UI-CustomTkinter-orange.svg)
![Status](https://img.shields.io/badge/Status-Open%20Source%20Preview-brightgreen.svg)
![Local AI](https://img.shields.io/badge/Local%20AI-Ollama%20%7C%20GGUF%20%7C%20HF-purple.svg)

> Made by Maomao for people who want to train small models with their own legal/local materials.  
> 猫猫制造：给想用自己的合法/本地资料训练小模型的人准备的训练集生产装置。

---

## What is this? / 这是什么？

**ForgeX NightCruise** is not a normal chatbot. It is a desktop tool for **systematically generating training datasets** from your own local documents by using a stronger teacher model.

**ForgeX NightCruise 不是普通聊天器。** 它是一个桌面工具，用来把你自己的本地文档、论文、课程材料、摘要、笔记或 JSON/TXT/PDF/DOCX 文件，通过更强的教师模型，系统性地转换成可用于训练小模型的结构化数据集。

In simple words:

```text
Local documents / 本地资料
        ↓
Teacher model / 教师模型
        ↓
Evidence-aligned bilingual training samples / 证据对齐的中英双语训练样本
        ↓
JSONL / JSON / TXT dataset / 可训练的数据集文件
        ↓
Fine-tune your own smaller model / 训练你自己的小模型
```

It is useful when you cannot or should not publicly collect large datasets, but you have local materials that you are allowed to process. NightCruise helps turn those materials into disciplined synthetic training data.

当你不能公开收集大量真实数据，或者不适合把资料上传到外部平台，但你手里有自己可以处理的本地资料时，NightCruise 就可以帮你把这些资料变成更规整、更适合训练的小模型数据。

---

## Core idea / 核心思想

NightCruise follows a **teacher-to-student data pipeline**:

1. You provide local materials.
2. A stronger teacher model reads the material.
3. NightCruise forces the teacher model to produce strict sections: overview, methods, author claims, claim-evidence alignment, limitations, reproduction requirements, and anti-hallucination QA.
4. Optional reviewer/validator models can critique or score the result.
5. The final output is saved as JSONL, JSON, or TXT for later fine-tuning.

NightCruise 采用的是 **教师模型 → 合成数据 → 学生模型** 的路线：

1. 你提供本地资料。
2. 强一些的教师模型阅读资料。
3. NightCruise 用严格提示词强制教师模型输出固定结构：研究概述、方法与数据、作者声明、声明-证据对齐、局限性、复现所需信息、反幻觉 QA。
4. 可选第二模型进行攻击式批判、幻觉检查或验证评分。
5. 最终导出为 JSONL / JSON / TXT，供后续 SFT、LoRA、QLoRA 或其他训练流程使用。

---

## The observation window / 观测窗：看见电脑是不是真的在干活

One of the most important details in NightCruise is the **GPU Control Center**, an always-visible observation window at the top of the application.

NightCruise 很重要的一个设计是顶部的 **GPU Control Center / 硬件观测窗**。它不是装饰品，而是为了让你实时知道电脑到底有没有在工作、负载在哪里、有没有过热风险。

It can show:

- **GPU Load / GPU 负载** — whether the GPU is really doing inference work.
- **VRAM usage / 显存占用** — whether the model/context is eating too much video memory.
- **Temperature / 温度** — helps you notice overheating early.
- **Power draw / 功耗** — useful for judging whether the GPU is actually under load.
- **CPU usage / CPU 占用** — tells you whether the job fell back to CPU or is bottlenecked there.
- **RAM usage / 内存占用** — helps avoid system swapping and crashes.
- **GPU clocks, memory clocks, fan speed, driver information / 核心频率、显存频率、风扇、驱动信息**.
- **CPU-only warning / CPU 模式警告** — warns you when no usable GPU is detected.

This means you can look at the observation window and quickly answer:

```text
Is my computer working?
Is the GPU working or only the CPU?
Is VRAM about to explode?
Is temperature too high?
Is the model stuck, idle, or actually generating?
```

也就是说，你可以直接看观测窗判断：

```text
电脑有没有真的在干活？
GPU 有没有跑起来，还是其实全在 CPU 上慢慢爬？
显存是不是快爆了？
温度是不是太高？
模型是在持续生成，还是已经卡住？
```

Important safety note: NightCruise can help you **observe** load and temperature, but it cannot physically replace good cooling, driver limits, power management, or common sense. If temperatures are abnormal, stop the job and check your hardware.

重要提醒：NightCruise 能帮助你**观察**负载和温度，提前发现异常，但它不能代替散热器、驱动保护、电源管理和人的判断。如果温度异常，请停止任务并检查硬件。

---

## Main features / 主要功能

### Dataset generation / 数据集生成

- Batch process TXT, Markdown, PDF, DOCX, JSON, JSONL, HTML documents.
- Extract abstract or first usable text block.
- Generate bilingual Chinese/English training samples.
- Enforce strict anti-hallucination structure.
- Export as JSONL, JSON, or TXT.

- 批量处理 TXT、Markdown、PDF、DOCX、JSON、JSONL、HTML 文档。
- 自动提取摘要或主要文本片段。
- 生成中英双语训练样本。
- 强制输出反幻觉结构。
- 支持 JSONL、JSON、TXT 导出。

### Multi-backend inference / 多后端推理

- Ollama local models.
- OpenAI-compatible API endpoints.
- llama.cpp server mode.
- Direct GGUF loading through `llama-cpp-python`.
- HuggingFace Transformers backend.

- 支持 Ollama 本地模型。
- 支持 OpenAI-compatible API。
- 支持 llama.cpp server 模式。
- 支持通过 `llama-cpp-python` 直接加载 GGUF。
- 支持 HuggingFace Transformers 后端。

### Fresh session optimization / 全新会话优化

NightCruise v7.1 is designed to avoid the common problem where long batch runs become slower and slower because of accumulated KV cache/context state.

NightCruise v7.1 针对批处理时速度越来越慢的问题做了优化：尽量让每个文件从新的推理会话开始，避免上下文/KV Cache 越堆越长。

- Ollama: uses chat endpoint and `keep_alive=0` cleanup.
- llama.cpp server: supports slot cache erase when enabled.
- GGUF: resets/clears model context when possible.
- HuggingFace: clears CUDA cache around generation.
- OpenAI-compatible: uses independent request sessions.

### Quality tools / 质量工具

- Quality guard for required sections.
- Optional validator model scoring.
- Optional second model review.
- Attack mode for aggressive critique.
- Hallucination defense mode.
- Deep thinking / quick mode.
- Generation history.
- Discipline classification and charts.
- Save/load configuration.

- 必填结构检查。
- 可选验证器模型评分。
- 可选第二模型复查。
- 攻击模式：更激进地挑错。
- 幻觉防御模式：专门检查无依据内容。
- 深度思考 / 快速模式。
- 生成历史记录。
- 学科分类与图表。
- 配置保存与加载。

---

## Minimum success standard / 最低成功标准

The first goal of NightCruise is not to create a miracle model. The first goal is to prove that the tool can **launch, keep running, process sample inputs, continuously produce readable structured output, and avoid meaningless garbled text**.

NightCruise 的第一目标不是制造神迹，而是证明它真的能动：**能启动、能持续运行、能处理样例输入、能有规律地产生可读结构化输出，而不是崩溃、卡死或吐出乱码。**

A minimal working run should satisfy:

- The UI launches.
- The observation window updates CPU/RAM/GPU state.
- A sample input file can be loaded.
- A teacher model can generate structured text.
- The output contains the required sections.
- The result can be saved to JSONL/JSON/TXT.

---

## Installation / 安装

Recommended Python version: **Python 3.10 or Python 3.11**.  
推荐 Python 版本：**Python 3.10 或 Python 3.11**。

```bash
git clone https://github.com/<your-name>/ForgeX-NightCruise.git
cd ForgeX-NightCruise
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
python selfcheck.py
python main.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python selfcheck.py
python main.py
```

You can also use the launcher scripts:

```bash
./launch.sh
```

Windows:

```bat
launch.bat
```

---

## Backend setup / 后端设置

### Option A: Ollama / 最简单路线

Install Ollama, pull a model, then set Provider to `ollama` in NightCruise.

```bash
ollama pull gemma2:2b
ollama serve
```

Default URL:

```text
http://localhost:11434
```

### Option B: OpenAI-compatible API / 兼容 API

Use any endpoint that supports OpenAI-style `/v1/chat/completions`. This can be a remote API or a local server.

### Option C: llama.cpp server / llama.cpp 服务器

If you use `llama-server`, enable **llama.cpp Server ★** in NightCruise so it can try to clear slot KV cache between generations.

```bash
llama-server -m your_model.gguf --host 0.0.0.0 --port 8080 --slots
```

NightCruise URL:

```text
http://localhost:8080/v1
```

### Option D: GGUF direct loading / 直接加载 GGUF

Install `llama-cpp-python` for your hardware. Examples:

```bash
pip install llama-cpp-python
```

CUDA 12.4:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Option E: HuggingFace Transformers / HF 模型

```bash
pip install transformers torch accelerate
```

Security note: only load HuggingFace models you trust. Some models require custom code.

安全提醒：只加载你信任的 HuggingFace 模型。有些模型需要自定义代码，存在供应链风险。

---

## Output format / 输出格式

JSONL example:

```json
{"text":"[CHINESE VERSION]...\n[ENGLISH VERSION]...","validation":{"score":10.0,"issues":[],"confidence":"high"},"discipline":{"major":"Computer Science","sub":"Artificial Intelligence"}}
```

TXT and JSON export are also supported.

也支持 TXT 和 JSON 导出。

---

## Project structure / 项目结构

```text
ForgeX-NightCruise/
├── main.py                    # Desktop UI entry point / 桌面 UI 入口
├── core/
│   ├── gpu_monitor.py         # CPU/RAM/GPU observation window backend / 观测窗后端
│   ├── inference.py           # Multi-backend inference engine / 多后端推理引擎
│   ├── text_processing.py     # Text extraction and classification / 文本提取与分类
│   └── prompts.py             # Strict dataset prompts / 严格数据集提示词
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_FORMAT.md
│   ├── MONITORING.md
│   └── SAFETY.md
├── examples/
│   ├── sample_input_paper.txt
│   └── sample_output.jsonl
├── requirements.txt
├── selfcheck.py
├── smoke_test.py
├── launch.bat / launch.sh
└── README.md
```

---

## Safety and data responsibility / 安全与数据责任

NightCruise is designed for local experimentation. You are responsible for the materials you process and the datasets you publish.

NightCruise 面向本地实验。你需要对自己处理的资料和发布的数据集负责。

Do not use it to process or publish materials you do not have the right to use. Do not upload private, confidential, medical, financial, or personal data to third-party APIs unless you fully understand the consequences.

不要用它处理或发布你无权使用的资料。不要把隐私、机密、医疗、金融或个人数据上传到第三方 API，除非你完全理解后果。

---

## Relationship with ForgeX Lab / 与 ForgeX Lab 的关系

**ForgeX Lab** is the model training/merging/export workbench.  
**ForgeX NightCruise** is the synthetic dataset factory.

**ForgeX Lab** 更像模型锻造工坊：训练、合并、导出模型。  
**ForgeX NightCruise** 更像训练矿石生产机：把本地资料变成训练集。

Together:

```text
NightCruise makes dataset ore.
ForgeX Lab forges models from that ore.

NightCruise 生产训练矿石。
ForgeX Lab 把矿石锻造成模型。
```

---

## License / 许可证

This project is released under the MIT License. See [LICENSE](LICENSE).

本项目使用 MIT License 开源。见 [LICENSE](LICENSE)。
