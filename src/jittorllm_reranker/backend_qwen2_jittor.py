"""JittorLLMs Qwen2.5 backend with deterministic generated-label scoring."""

from __future__ import annotations

import gc
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    text: str
    runtime_sec: float
    generated_token_ids: list[int]


@dataclass
class LabelScoreResult:
    score: float
    raw_output: str
    runtime_sec: float
    selected_token_id: int
    negative_logit: float
    positive_logit: float


class Qwen2JittorBackend:
    """Load JittorLLMs' Qwen2 implementation and run greedy CUDA inference."""

    CHAT_TEMPLATE = (
        "<|im_start|>system\n"
        "You are a relevance classifier. Follow the requested output format exactly."
        "<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(
        self,
        model_path: str | Path,
        jittorllms_root: str | Path,
        max_input_length: int = 256,
        max_new_tokens: int = 4,
    ) -> None:
        import jittor as jt

        self.jt = jt
        self.model_path = Path(model_path).resolve()
        self.jittorllms_root = Path(jittorllms_root).resolve()
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self.backend_name = "qwen2_jittorllms"
        self.model_name = Path(model_path).name

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        qwen2_root = self.jittorllms_root / "models" / "qwen2"
        if not qwen2_root.is_dir():
            raise FileNotFoundError(f"JittorLLMs Qwen2 directory not found: {qwen2_root}")
        checkpoint = self.model_path / "model.fp16.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Converted checkpoint not found: {checkpoint}. "
                "Run scripts/convert_qwen2_safetensors_to_pth.py first."
            )

        jt.flags.use_cuda = 1
        if not jt.has_cuda:
            raise RuntimeError("Jittor CUDA is unavailable; refusing to run Qwen2 on CPU")

        sys.path.insert(0, str(qwen2_root))
        modeling = importlib.import_module("qwen2_jt.modeling_qwen2")
        tokenization = importlib.import_module("qwen2_jt.tokenization_qwen2_fast")

        config_data = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        config = modeling.Qwen2Config(**config_data)
        self.model = modeling.Qwen2ForCausalLM(config=config)
        self.model.half()
        state_dict = jt.load(str(checkpoint))
        self.model.load_state_dict(state_dict)
        del state_dict
        if config_data.get("tie_word_embeddings", False):
            self.model.lm_head.weight = self.model.model.embed_tokens.weight
        self.model.eval()
        jt.gc()
        jt.sync_all()

        tokenizer_path = self.model_path / "tokenizer.json"
        self.tokenizer = tokenization.Qwen2TokenizerFast(tokenizer_file=str(tokenizer_path))
        self.eos_token_ids = {151643, 151645}

    def _encode(self, prompt: str) -> tuple[list[int], list[int]]:
        rendered = self.CHAT_TEMPLATE.format(prompt=prompt)
        encoded = self.tokenizer._batch_encode_plus([rendered])
        input_ids = list(encoded["input_ids"][0])
        if len(input_ids) > self.max_input_length:
            tail = min(64, self.max_input_length // 3)
            input_ids = input_ids[: self.max_input_length - tail] + input_ids[-tail:]
        return input_ids, [1] * len(input_ids)

    def score_binary_tokens(self, prompt: str) -> LabelScoreResult:
        """Return logit("1") - logit("0") from one real Jittor forward pass."""

        jt = self.jt
        input_ids_list, attention_mask_list = self._encode(prompt)
        input_ids = jt.array([input_ids_list], dtype=jt.int64)
        attention_mask = jt.array([attention_mask_list], dtype=jt.int64)
        label_ids = {
            label: self.tokenizer._batch_encode_plus([label])["input_ids"][0]
            for label in ("0", "1")
        }
        if any(len(ids) != 1 for ids in label_ids.values()):
            raise RuntimeError(f"Expected single-token numeric labels, got: {label_ids}")

        started = time.perf_counter()
        with jt.no_grad():
            position_ids = attention_mask.long().cumsum(-1) - 1
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                num_logits_to_keep=1,
            )[0][0, -1, :].float()
            selected = logits[jt.array([label_ids["0"][0], label_ids["1"][0]])]
            values = selected.numpy().tolist()
        jt.sync_all()
        runtime = time.perf_counter() - started
        negative_logit, positive_logit = float(values[0]), float(values[1])
        raw_output = "1" if positive_logit >= negative_logit else "0"
        token_id = label_ids[raw_output][0]
        return LabelScoreResult(
            score=positive_logit - negative_logit,
            raw_output=raw_output,
            runtime_sec=runtime,
            selected_token_id=token_id,
            negative_logit=negative_logit,
            positive_logit=positive_logit,
        )

    def generate(self, prompt: str) -> GenerationResult:
        jt = self.jt
        input_ids_list, attention_mask_list = self._encode(prompt)
        input_ids = jt.array([input_ids_list], dtype=jt.int64)
        attention_mask = jt.array([attention_mask_list], dtype=jt.int64)
        generated: list[int] = []
        started = time.perf_counter()

        with jt.no_grad():
            for _ in range(self.max_new_tokens):
                position_ids = attention_mask.long().cumsum(-1) - 1
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    num_logits_to_keep=1,
                )
                next_token, _ = jt.argmax(outputs[0][:, -1, :].float(), dim=-1)
                token_id = int(next_token.numpy()[0])
                if token_id in self.eos_token_ids:
                    break
                generated.append(token_id)
                input_ids = jt.concat([input_ids, next_token.reshape(1, 1).int64()], dim=-1)
                attention_mask = jt.concat(
                    [attention_mask, jt.ones((1, 1), dtype=jt.int64)], dim=-1
                )
                del outputs

        jt.sync_all()
        runtime = time.perf_counter() - started
        text = self.tokenizer._decode(generated, skip_special_tokens=True)
        return GenerationResult(text=text, runtime_sec=runtime, generated_token_ids=generated)

    def close(self) -> None:
        del self.model
        gc.collect()
        self.jt.gc()
