import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..env.strands import StrandsEnv
from ..data.dataset import StrandsDataset

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class StrandsBenchmarkResults:
    fully_solved_count: int            # puzzles where all theme words found
    total: int
    theme_words_found_counts: list     # theme words found per puzzle
    total_theme_words_counts: list     # total theme words per puzzle
    scores: list                       # total reward per puzzle

    @property
    def solve_rate(self) -> float:
        return self.fully_solved_count / self.total if self.total > 0 else 0.0

    @property
    def avg_theme_words_found_pct(self) -> float:
        if not self.theme_words_found_counts:
            return 0.0
        fracs = [f / t for f, t in zip(self.theme_words_found_counts, self.total_theme_words_counts) if t > 0]
        return sum(fracs) / len(fracs) if fracs else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def print_summary(self):
        print("=" * 50)
        print("Strands Benchmark Results")
        print("=" * 50)
        print(f"Puzzles evaluated:      {self.total}")
        print(f"Fully solved:           {self.fully_solved_count}/{self.total} ({self.solve_rate:.1%})")
        print(f"Avg theme words found:  {self.avg_theme_words_found_pct:.1%}")
        print(f"Avg score:              {self.avg_score:.1f}")
        print("=" * 50)

    def save(self, path: str):
        data = {
            "fully_solved_count": self.fully_solved_count,
            "total": self.total,
            "solve_rate": self.solve_rate,
            "avg_theme_words_found_pct": self.avg_theme_words_found_pct,
            "avg_score": self.avg_score,
            "per_puzzle_scores": self.scores,
            "per_puzzle_theme_words_found": self.theme_words_found_counts,
            "per_puzzle_total_theme_words": self.total_theme_words_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


class StrandsBenchmark:
    """
    Evaluates a model over the StrandsDataset benchmark.

    Usage:
        benchmark = StrandsBenchmark(model_path='./model')
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('strands_results.json')
    """

    def __init__(
        self,
        model_path: str,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            model_path = Path(model_path)
            adapter_config_path = model_path / "adapter_config.json"

            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path", adapter_cfg.get("base_model"))
                print(f"Loading base model: {base_name}")
                base = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )
                self.model = PeftModel.from_pretrained(base, str(model_path))
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                )

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()

        self.dataset = StrandsDataset()
        self._system_prompt = (_PROMPTS_DIR / "strands_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "strands_user.md").read_text().strip()

    def _generate_guess(self, messages: list, temperature: float = 0.0, max_new_tokens: int = 10) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
        raw = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        word = raw.split()[0] if raw.split() else raw
        return "".join(c for c in word if c.isalpha()).upper()

    def run(
        self,
        num_puzzles: Optional[int] = None,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> StrandsBenchmarkResults:
        """
        Play full multi-turn Strands games over the dataset.

        Args:
            num_puzzles: number of puzzles to evaluate (None = all)
            temperature: 0.0 = greedy decoding (recommended for eval)
            verbose: print progress every 50 puzzles
        """
        n = min(num_puzzles, len(self.dataset)) if num_puzzles else len(self.dataset)
        fully_solved_count = 0
        theme_words_found_counts = []
        total_theme_words_counts = []
        scores = []

        for idx in range(n):
            item = self.dataset[idx]
            config = self.dataset.get_config(item["puzzle_id"])
            env = StrandsEnv(config)
            obs, _ = env.reset()

            max_guesses = config.max_guesses  # 3 × num_theme_words by default
            num_theme_words = len(config.theme_words)

            user_msg = self._user_prompt_template.format(
                theme=config.theme,
                num_theme_words=num_theme_words,
                max_guesses=max_guesses,
                board_str=obs["board_str"],
            )
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]

            terminated = False
            truncated = False
            for _ in range(max_guesses):
                guess = self._generate_guess(messages, temperature)
                messages.append({"role": "assistant", "content": guess})
                obs, reward, terminated, truncated, _ = env.step(guess)

                if terminated or truncated:
                    break

                # Feedback message (mirror LLMHandler._strands_feedback)
                feedback = obs.get("feedback", "")
                if reward == 5:
                    fb = f"'{guess}': {feedback} Spanagram! +5 points."
                elif reward == 1:
                    fb = f"'{guess}': {feedback} +1 point."
                else:
                    fb = f"'{guess}': {feedback}"
                fb += (
                    f"\n{obs['progress']}"
                    f"\nGuesses used: {obs['num_guesses']}."
                    f"\n\nUpdated board:\n{obs['board_str']}"
                    f"\nPlease guess another word."
                )
                messages.append({"role": "user", "content": fb})

            if terminated:
                fully_solved_count += 1

            theme_words_found_counts.append(len(obs["theme_words_guessed"]))
            total_theme_words_counts.append(num_theme_words)
            total_reward = sum(r for _, r, _ in env.info.get("history", []))
            scores.append(total_reward)

            if verbose and (idx + 1) % 50 == 0:
                rate = fully_solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Fully solved: {fully_solved_count}/{idx + 1} ({rate:.1%})")

        return StrandsBenchmarkResults(
            fully_solved_count=fully_solved_count,
            total=n,
            theme_words_found_counts=theme_words_found_counts,
            total_theme_words_counts=total_theme_words_counts,
            scores=scores,
        )
