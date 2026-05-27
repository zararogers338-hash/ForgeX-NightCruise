# Safety Notes / 安全说明

## Data privacy / 数据隐私

NightCruise can work with local files, but some inference backends may send text to remote APIs. Before using OpenAI-compatible remote endpoints, check whether your documents contain private or confidential data.

NightCruise 可以处理本地文件，但部分推理后端可能会把文本发送到远程 API。在使用 OpenAI-compatible 远程接口前，请确认你的文件是否包含隐私或机密数据。

## Model safety / 模型安全

Only load local GGUF or HuggingFace models from trusted sources. Some model loading paths can execute model-specific code depending on the backend.

只加载来自可信来源的 GGUF 或 HuggingFace 模型。部分模型加载方式可能会执行模型自带代码，取决于后端。

## Hardware safety / 硬件安全

The monitoring window helps you observe GPU/CPU/RAM/temperature state, but it does not replace good cooling or driver-level protections.

观测窗能帮助你观察 GPU/CPU/RAM/温度状态，但不能替代良好散热和驱动级保护。
