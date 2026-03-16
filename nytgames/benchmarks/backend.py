"""
Backend abstraction for NYT Games benchmarks.

Provides HFBackend (local HuggingFace model) and CloudBackend (OpenAI-compatible API),
both implementing the GuessBackend protocol.

Key utilities:
    extract_answer(text)               -- parse "Answer:" sentinel from thinking-format output
    generate_from_candidates(...)      -- pick best candidate via logprobs (HF) or shortlist prompt (cloud)
"""

import re
from typing import Protocol, runtime_checkable

_THINKING_SUFFIX = (
    "\n\nThink step-by-step inside <think>...</think> tags before answering. "
    "After your thinking, write \"Answer:\" followed by your final answer on a new line."
)


def _apply_thinking_prompt(system_content: str) -> str:
    return system_content + _THINKING_SUFFIX


def extract_answer(text: str) -> str:
    """Extract the answer from thinking-format output.

    Looks for "Answer: ..." sentinel; falls back to the last non-empty line.
    """
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text


def generate_from_candidates(backend: "GuessBackend", messages: list, candidates: list[str]) -> str:
    """Pick the best candidate word.

    For HFBackend: scores each candidate by log-probability and returns the argmax.
    For CloudBackend (score_candidates returns None): asks the model to choose from
    a shortlist of up to 5 candidates.
    """
    scores = backend.score_candidates(messages, candidates)
    if scores is not None:
        return candidates[scores.index(max(scores))]

    # Cloud fallback: ask the model to pick from a shortlist
    shortlist = candidates[:5]
    shortlist_msg = messages + [{
        "role": "user",
        "content": (
            f"Choose the single best next guess from these options: {', '.join(shortlist)}. "
            "Reply with only the chosen word, nothing else."
        ),
    }]
    raw = backend.generate_text(shortlist_msg, max_new_tokens=16, temperature=0.0)
    chosen = extract_answer(raw) if backend.use_thinking_format else (raw.split()[0] if raw.split() else raw)
    chosen = "".join(c for c in chosen if c.isalpha()).upper()
    return chosen if chosen in shortlist else shortlist[0]


@runtime_checkable
class GuessBackend(Protocol):
    use_thinking_format: bool

    def generate_text(self, messages: list, *, max_new_tokens: int, temperature: float) -> str:
        ...

    def score_candidates(self, messages: list, candidates: list[str]) -> list[float] | None:
        ...


class HFBackend:
    """Local HuggingFace model backend with logprob-based candidate scoring."""

    def __init__(self, model, tokenizer, use_thinking_format: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_thinking_format = use_thinking_format

    def generate_text(self, messages: list, *, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        import torch

        msgs = list(messages)
        if self.use_thinking_format and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "system", "content": _apply_thinking_prompt(msgs[0]["content"])}] + msgs[1:]

        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def score_candidates(self, messages: list, candidates: list[str]) -> list[float]:
        """Score each candidate string by summing log-probs of its tokens."""
        import torch
        import torch.nn.functional as F

        msgs = list(messages)
        if self.use_thinking_format and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "system", "content": _apply_thinking_prompt(msgs[0]["content"])}] + msgs[1:]

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.model.device)
        len_prompt = prompt_ids.shape[1]

        scores = []
        with torch.no_grad():
            for candidate in candidates:
                cand_ids = self.tokenizer(
                    candidate, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.model.device)

                full_ids = torch.cat([prompt_ids, cand_ids], dim=1)
                logits = self.model(input_ids=full_ids).logits[0]  # [seq_len, vocab]
                log_probs = F.log_softmax(logits, dim=-1)

                score = 0.0
                for i, tok_id in enumerate(cand_ids[0]):
                    # logit at position (len_prompt - 1 + i) predicts token at (len_prompt + i)
                    score += log_probs[len_prompt - 1 + i, tok_id].item()
                scores.append(score)

        return scores


class CloudBackend:
    """OpenAI-compatible cloud API backend (no logprob scoring)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://tritonai-api.ucsd.edu",
        use_thinking_format: bool = False,
    ):
        import os
        from openai import OpenAI

        self.model = model
        self.use_thinking_format = use_thinking_format
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)

    def generate_text(self, messages: list, *, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        msgs = list(messages)
        if self.use_thinking_format and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "system", "content": _apply_thinking_prompt(msgs[0]["content"])}] + msgs[1:]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content or ""

    def score_candidates(self, messages: list, candidates: list[str]) -> None:
        # Cloud APIs don't expose per-token log-probs in the required form
        return None
