# Installation / 安装说明

## Recommended environment / 推荐环境

- Python 3.10 or 3.11
- Windows, Linux, or macOS
- Ollama, llama.cpp server, GGUF, HuggingFace, or OpenAI-compatible endpoint
- NVIDIA GPU is optional but recommended for local inference

## Quick install / 快速安装

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scriptsctivate
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

## Smoke test / 冒烟测试

```bash
python smoke_test.py
```

The smoke test does not run a real LLM. It checks the basic document-processing and output-format logic.

冒烟测试不会运行真实大模型，只检查基础文档处理与输出格式逻辑。
