# -*- coding: utf-8 -*-
"""
Inference Engine - Multi-backend LLM inference with GPU acceleration
Supports: Ollama, OpenAI-compatible, llama-cpp-python (GGUF), HuggingFace Transformers

v7.1 — Fresh Session Optimization:
  每次流式生成都会开启全新的 API 会话窗口，
  彻底清除上一次推理的 KV Cache，防止 t/s 持续下降。

  ★ 特别支持 llama.cpp server (llama-server):
    - 通过 /slots API 擦除 slot KV Cache
    - 请求中附加 cache_prompt=false 禁用前缀缓存复用
    - 需勾选 UI 中的 "llama.cpp Server ★" 复选框启用
"""

import json
import os
import time
import threading
import gc
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    provider: str = "ollama"  # ollama, openai_compatible, local_gguf, huggingface
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    gguf_path: str = ""
    hf_model_id: str = ""
    model: str = ""
    n_ctx: int = 8192
    n_gpu_layers: int = -1  # -1 = auto, 0 = CPU only
    n_batch: int = 512
    n_threads: int = 0  # 0 = auto
    rope_freq_base: float = 0.0
    rope_freq_scale: float = 0.0
    flash_attention: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    offload_kqv: bool = True
    tensor_split: Optional[List[float]] = None
    seed: int = -1
    verbose: bool = False
    is_llama_server: bool = False  # ★ llama.cpp server 模式 (slot KV Cache 管理)


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    system_prompt: str = ""
    max_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    grammar: Optional[str] = None  # GBNF grammar for structured output


@dataclass
class InferenceMetrics:
    """Performance metrics for inference"""
    tokens_generated: int = 0
    total_time_s: float = 0.0
    prompt_eval_time_s: float = 0.0
    generation_time_s: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    peak_vram_mb: float = 0.0
    backend_used: str = "unknown"
    session_id: int = 0  # 会话序号，用于追踪是否为新会话


# 全局会话计数器
_session_counter = 0


def _next_session_id() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


class InferenceEngine:
    """
    Industrial-grade inference engine with multi-backend support.

    v7.1 Fresh Session 优化:
    - Ollama:  每次生成后发送 keep_alive=0 强制卸载模型上下文，
               下次调用时 Ollama 会重新加载模型并分配全新的 KV Cache
    - GGUF:   每次生成前 reset() 清除 llama.cpp 内部 KV Cache
    - HF:     每次生成后 torch.cuda.empty_cache() 释放显存碎片
    - OpenAI: 每次使用独立的 requests.Session，生成后立即关闭
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model_cache = {}
        self._lock = threading.Lock()
        self._current_metrics = InferenceMetrics()

    def _get_n_threads(self) -> int:
        if self.config.n_threads > 0:
            return self.config.n_threads
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 1)

    def _load_gguf_model(self, gguf_path: str, force_gpu: bool = False):
        """Load GGUF model via llama-cpp-python"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed.\n"
                "Install with GPU: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
                "Install CPU only: pip install llama-cpp-python"
            )

        if not gguf_path or not Path(gguf_path).exists():
            raise RuntimeError(f"GGUF file not found: {gguf_path}")

        cache_key = f"gguf:{gguf_path}:{self.config.n_gpu_layers}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        n_gpu_layers = self.config.n_gpu_layers
        if force_gpu:
            n_gpu_layers = -1  # Offload all layers to GPU
        elif n_gpu_layers == -1:
            n_gpu_layers = 999  # Auto = try all layers

        params = {
            "model_path": gguf_path,
            "n_ctx": self.config.n_ctx,
            "n_threads": self._get_n_threads(),
            "n_batch": self.config.n_batch,
            "n_gpu_layers": n_gpu_layers,
            "use_mmap": self.config.use_mmap,
            "use_mlock": self.config.use_mlock,
            "offload_kqv": self.config.offload_kqv,
            "flash_attn": self.config.flash_attention,
            "verbose": self.config.verbose,
        }

        if self.config.rope_freq_base > 0:
            params["rope_freq_base"] = self.config.rope_freq_base
        if self.config.rope_freq_scale > 0:
            params["rope_freq_scale"] = self.config.rope_freq_scale
        if self.config.seed >= 0:
            params["seed"] = self.config.seed
        if self.config.tensor_split:
            params["tensor_split"] = self.config.tensor_split

        llm = Llama(**params)
        self._model_cache[cache_key] = llm
        return llm

    def _load_hf_model(self, model_id: str, force_gpu: bool = False):
        """Load HuggingFace model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise RuntimeError(
                "transformers is not installed.\n"
                "pip install transformers torch accelerate"
            )

        cache_key = f"hf:{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Determine device
        if force_gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                raise RuntimeError("No GPU available! Cannot force GPU inference.")
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            pass  # Already on CPU
        elif device in ("cuda", "mps") and not hasattr(model, 'hf_device_map'):
            model = model.to(device)

        self._model_cache[cache_key] = (model, tokenizer, device)
        return model, tokenizer, device

    # ------------------------------------------------------------------
    #  核心: 每次生成都开启全新会话
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        user_prompt: str,
        gen_config: GenerationConfig,
        callback: Callable[[str], None],
        force_gpu: bool = False,
        is_validator: bool = False,
    ) -> tuple:
        """
        Stream generate text. Returns (full_text, metrics).

        v7.1: 每次调用都会:
           1. 为当前请求分配新的 session_id
           2. 生成前清除上一次的 KV Cache
           3. 生成后立即释放本次会话的服务端上下文
        """
        session_id = _next_session_id()
        start_time = time.time()
        full_text = ""
        metrics = InferenceMetrics(session_id=session_id)

        effective_temp = 0.0 if is_validator else gen_config.temperature
        effective_max_tokens = 512 if is_validator else gen_config.max_tokens

        provider = self.config.provider

        try:
            if provider == "local_gguf":
                full_text, metrics = self._generate_gguf(
                    user_prompt, gen_config, effective_temp, effective_max_tokens,
                    callback, force_gpu, session_id
                )
            elif provider == "huggingface":
                full_text, metrics = self._generate_hf(
                    user_prompt, gen_config, effective_temp, effective_max_tokens,
                    callback, force_gpu, session_id
                )
            elif provider == "ollama":
                full_text, metrics = self._generate_ollama(
                    user_prompt, gen_config, effective_temp, effective_max_tokens,
                    callback, session_id
                )
            elif provider == "openai_compatible":
                full_text, metrics = self._generate_openai(
                    user_prompt, gen_config, effective_temp, effective_max_tokens,
                    callback, session_id
                )
            else:
                raise RuntimeError(f"Unknown provider: {provider}")

        except Exception as e:
            raise RuntimeError(f"[{provider}] Inference error: {e}")

        metrics.total_time_s = time.time() - start_time
        metrics.tokens_per_second = (
            metrics.tokens_generated / metrics.generation_time_s
            if metrics.generation_time_s > 0 else 0
        )
        metrics.session_id = session_id
        self._current_metrics = metrics
        return full_text, metrics

    # ------------------------------------------------------------------
    #  GGUF Backend (llama-cpp-python)
    # ------------------------------------------------------------------

    def _generate_gguf(self, user_prompt, gen_config, temp, max_tokens,
                       callback, force_gpu, session_id):
        llm = self._load_gguf_model(self.config.gguf_path, force_gpu)
        metrics = InferenceMetrics(backend_used="llama.cpp", session_id=session_id)

        # ★ 生成前: 重置 KV Cache，确保全新上下文
        self._reset_gguf_kv_cache(llm)

        gen_start = time.time()
        stream = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": gen_config.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            stream=True,
            temperature=temp,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            min_p=gen_config.min_p,
            repeat_penalty=gen_config.repeat_penalty,
            presence_penalty=gen_config.presence_penalty,
            frequency_penalty=gen_config.frequency_penalty,
        )

        full = ""
        token_count = 0
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full += token
                token_count += 1
                callback(token)

        metrics.tokens_generated = token_count
        metrics.generation_time_s = time.time() - gen_start

        # ★ 生成后: 再次清除 KV Cache，释放显存
        self._reset_gguf_kv_cache(llm)

        return full, metrics

    def _reset_gguf_kv_cache(self, llm):
        """清除 llama.cpp 模型的 KV Cache，让下次推理从全新状态开始"""
        try:
            # llama-cpp-python >= 0.2.x 支持 reset() 方法
            if hasattr(llm, 'reset'):
                llm.reset()
            # 部分版本通过 _ctx 直接操作
            elif hasattr(llm, '_ctx') and llm._ctx is not None:
                try:
                    from llama_cpp import llama_cpp
                    if hasattr(llama_cpp, 'llama_kv_cache_clear'):
                        llama_cpp.llama_kv_cache_clear(llm._ctx.ctx)
                except Exception:
                    pass
            # 兜底: 重置内部 token 计数器
            if hasattr(llm, 'n_tokens'):
                llm.n_tokens = 0
            if hasattr(llm, '_input_ids'):
                llm._input_ids = []
        except Exception:
            pass  # 静默处理，不影响主流程

    # ------------------------------------------------------------------
    #  HuggingFace Backend
    # ------------------------------------------------------------------

    def _generate_hf(self, user_prompt, gen_config, temp, max_tokens,
                     callback, force_gpu, session_id):
        """Generate using HuggingFace transformers"""
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError:
            raise RuntimeError("transformers/torch not installed")

        model, tokenizer, device = self._load_hf_model(
            self.config.hf_model_id or self.config.model, force_gpu
        )
        metrics = InferenceMetrics(backend_used=f"hf:{device}", session_id=session_id)

        # ★ 生成前: 清理 GPU 显存碎片，确保有足够空间给新的 KV Cache
        self._flush_torch_cache(device)

        # Build prompt
        messages = [
            {"role": "system", "content": gen_config.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(tokenizer, 'apply_chat_template'):
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = f"{gen_config.system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt")
        if device != "cpu" and not hasattr(model, 'hf_device_map'):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(temp, 0.01),
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "repetition_penalty": gen_config.repeat_penalty,
            "do_sample": temp > 0,
            "streamer": streamer,
        }

        gen_start = time.time()
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        full = ""
        token_count = 0
        for text in streamer:
            if text:
                full += text
                token_count += 1
                callback(text)

        thread.join()
        metrics.tokens_generated = token_count
        metrics.generation_time_s = time.time() - gen_start

        # ★ 生成后: 释放中间张量和 KV Cache 占用的显存
        self._flush_torch_cache(device)

        return full, metrics

    def _flush_torch_cache(self, device: str):
        """清除 PyTorch GPU 缓存，释放 KV Cache 碎片"""
        gc.collect()
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, 'mps'):
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Ollama Backend — ★ 核心优化: 全新 API 会话窗口
    # ------------------------------------------------------------------

    def _generate_ollama(self, user_prompt, gen_config, temp, max_tokens,
                         callback, session_id):
        """
        Ollama 流式生成 — 每次都开新会话窗口

        优化策略:
        1. 使用 /api/chat 接口（而非 /api/generate），每次传入干净的 messages 数组
        2. 不传递上一次的 context token，确保 KV Cache 从零开始
        3. 生成完毕后发送 keep_alive=0 强制卸载模型上下文
        4. 使用独立的 requests.Session 并在完成后关闭
        """
        import requests
        metrics = InferenceMetrics(backend_used="ollama", session_id=session_id)

        base = self.config.base_url.rstrip('/')

        # ★ Step 1: 生成前先刷掉旧的 KV Cache
        self._ollama_flush_context(base, self.config.model)

        # ★ Step 2: 使用全新的 Session 发起请求
        session = requests.Session()
        try:
            # 使用 /api/chat 接口 — 每次传全新的 messages，不带历史 context
            url = f"{base}/api/chat"
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": gen_config.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temp,
                    "top_p": gen_config.top_p,
                    "top_k": gen_config.top_k,
                    "repeat_penalty": gen_config.repeat_penalty,
                    "num_ctx": self.config.n_ctx,        # 显式指定上下文窗口大小
                    "num_batch": self.config.n_batch,    # 显式指定 batch size
                },
                # ★ 关键: 不传 context 字段 -> Ollama 会分配全新的 KV Cache
            }

            gen_start = time.time()
            full = ""
            token_count = 0

            with session.post(url, json=payload, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        # /api/chat 接口的 token 在 message.content 中
                        msg = data.get("message", {})
                        token = msg.get("content", "")
                        if token:
                            full += token
                            token_count += 1
                            callback(token)
                        if data.get("done"):
                            # 提取 Ollama 性能指标
                            metrics.prompt_tokens = data.get("prompt_eval_count", 0)
                            eval_dur = data.get("eval_duration", 0)
                            if eval_dur > 0:
                                metrics.generation_time_s = eval_dur / 1e9

            if metrics.generation_time_s == 0:
                metrics.generation_time_s = time.time() - gen_start
            metrics.tokens_generated = token_count

        finally:
            # ★ Step 3: 关闭 Session，断开连接
            session.close()

        # ★ Step 4: 生成后立即刷掉本次的 KV Cache
        self._ollama_flush_context(base, self.config.model)

        return full, metrics

    def _ollama_flush_context(self, base_url: str, model_name: str):
        """
        向 Ollama 发送 keep_alive=0 请求，强制卸载模型的 KV Cache。
        下次请求时 Ollama 会重新加载模型并分配全新的上下文窗口。

        这等效于 "在服务器中打开一个全新的 API 窗口"。
        """
        import requests
        try:
            # 通过 /api/generate 发送空请求 + keep_alive=0
            requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,  # 0 = 立即卸载
                },
                timeout=10
            )
        except Exception:
            pass  # 网络错误时静默处理，不影响主流程

        # 短暂等待，确保 Ollama 完成上下文释放
        time.sleep(0.3)

    # ------------------------------------------------------------------
    #  OpenAI-Compatible Backend (含 llama.cpp server 专项优化)
    # ------------------------------------------------------------------

    def _generate_openai(self, user_prompt, gen_config, temp, max_tokens,
                         callback, session_id):
        """
        OpenAI-compatible 流式生成

        ★ llama.cpp server 专项优化:
        - 自动检测或手动指定 is_llama_server 模式
        - 生成前清除 slot 的 KV Cache (/slots API)
        - 请求中附加 cache_prompt: false 禁止前缀缓存复用
        - 生成后再次清除 slot
        - 每次使用独立 Session 并关闭
        """
        import requests
        is_llama = self.config.is_llama_server
        backend_label = "llama.cpp-server" if is_llama else "openai_compatible"
        metrics = InferenceMetrics(backend_used=backend_label, session_id=session_id)

        base = self.config.base_url.rstrip('/')

        # ★ llama.cpp server: 生成前清 slot KV Cache
        if is_llama:
            self._llamaserver_flush_slots(base)

        url = f"{base}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": gen_config.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": gen_config.top_p,
            "presence_penalty": gen_config.presence_penalty,
            "frequency_penalty": gen_config.frequency_penalty,
        }

        # ★ llama.cpp server: 禁用 prompt cache，强制从头计算
        if is_llama:
            payload["cache_prompt"] = False
            # 部分 llama.cpp 版本支持直接在 payload 指定 slot_id
            payload["slot_id"] = -1  # -1 = 自动分配新 slot

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # ★ 每次使用全新的 Session
        session = requests.Session()
        try:
            gen_start = time.time()
            full = ""
            token_count = 0

            with session.post(url, json=payload, headers=headers, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            line_str = line_str[6:]
                        if line_str == "[DONE]":
                            break
                        try:
                            data = json.loads(line_str)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                full += token
                                token_count += 1
                                callback(token)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            metrics.tokens_generated = token_count
            metrics.generation_time_s = time.time() - gen_start

        finally:
            # ★ 立即关闭 Session，释放连接
            session.close()

        # ★ llama.cpp server: 生成后再清 slot KV Cache
        if is_llama:
            self._llamaserver_flush_slots(base)

        return full, metrics

    def _llamaserver_flush_slots(self, base_url: str):
        """
        清除 llama.cpp server 的 slot KV Cache。

        llama.cpp server 用 "slot" 管理 KV Cache:
        - 默认只有 1 个 slot (--parallel 1)
        - 每个 slot 有独立的 KV Cache 和 prompt cache
        - 如果不清理，上一次请求的 KV Cache 会残留在 slot 中

        清理策略 (按优先级尝试):
        1. POST /slots/0?action=erase  — 直接擦除 slot 0 的 KV Cache
        2. POST /slots/{id}?action=erase — 擦除所有 slot
        3. 发送一个极短的空请求冲掉旧缓存 (兜底)

        注意: /slots API 需要 llama-server 启动时带 --slots 参数
        """
        import requests

        # 方法1: 尝试列出所有 slots 并逐一擦除
        try:
            r = requests.get(f"{base_url}/slots", timeout=5)
            if r.status_code == 200:
                slots = r.json()
                for slot in slots:
                    slot_id = slot.get("id", 0)
                    try:
                        requests.post(
                            f"{base_url}/slots/{slot_id}?action=erase",
                            timeout=5
                        )
                    except Exception:
                        pass
                time.sleep(0.1)
                return
        except Exception:
            pass

        # 方法2: 直接尝试擦除 slot 0 (最常见场景)
        try:
            r = requests.post(f"{base_url}/slots/0?action=erase", timeout=5)
            if r.status_code == 200:
                time.sleep(0.1)
                return
        except Exception:
            pass

        # 方法3: 兜底 — 发一个极短的空请求冲掉旧缓存
        try:
            requests.post(
                f"{base_url}/completion",
                json={
                    "prompt": " ",
                    "n_predict": 1,
                    "cache_prompt": False,
                },
                timeout=10
            )
        except Exception:
            pass

        time.sleep(0.2)

    # ------------------------------------------------------------------
    #  模型管理
    # ------------------------------------------------------------------

    def reset_context(self):
        """
        重置所有后端的上下文 (KV Cache)，不卸载模型权重。
        比 unload_models() 轻量，只清缓存不释放模型。
        """
        provider = self.config.provider

        if provider == "ollama":
            base = self.config.base_url.rstrip('/')
            self._ollama_flush_context(base, self.config.model)

        elif provider == "openai_compatible" and self.config.is_llama_server:
            # ★ llama.cpp server: 清除 slot KV Cache
            base = self.config.base_url.rstrip('/')
            self._llamaserver_flush_slots(base)

        elif provider == "local_gguf":
            for key, val in self._model_cache.items():
                if key.startswith("gguf:"):
                    self._reset_gguf_kv_cache(val)

        elif provider == "huggingface":
            for key, val in self._model_cache.items():
                if key.startswith("hf:"):
                    _, _, device = val
                    self._flush_torch_cache(device)

        gc.collect()

    def unload_models(self):
        """Free all cached models from memory"""
        # 先清上下文
        self.reset_context()

        for key, val in self._model_cache.items():
            if key.startswith("gguf:"):
                try:
                    del val
                except Exception:
                    pass
            elif key.startswith("hf:"):
                try:
                    model, tokenizer, device = val
                    del model
                    del tokenizer
                except Exception:
                    pass

        self._model_cache.clear()

        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    @property
    def last_metrics(self) -> InferenceMetrics:
        return self._current_metrics


def get_models(provider: str, base_url: str, api_key: str) -> List[str]:
    """Fetch available models from provider"""
    base_url = base_url.rstrip("/")
    if provider == "ollama":
        try:
            import requests
            r = requests.get(f"{base_url}/api/tags", timeout=10)
            if r.status_code == 200:
                return sorted([m["name"] for m in r.json().get("models", [])])
        except Exception:
            return ["Error: Ollama connection failed"]
    elif provider == "openai_compatible":
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = requests.get(f"{base_url}/models", headers=headers, timeout=10)
            if r.status_code == 200:
                return sorted([m["id"] for m in r.json().get("data", [])])
            return [f"HTTP {r.status_code}"]
        except Exception:
            return ["Error: API connection failed"]
    elif provider == "huggingface":
        return ["(Enter HF model ID manually)"]
    elif provider == "local_gguf":
        return ["(Select GGUF file)"]
    return ["No models"]
