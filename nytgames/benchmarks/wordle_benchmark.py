import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..env.wordle import WordleEnv, WordleConfig, WORDLE_WORD_LENGTH
from ..data.dataset import WordleDataset, load_dictionary

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class BenchmarkResults:
    solved_count: int
    total: int
    guesses_used: list  # guesses used for solved puzzles only
    scores: list        # total_points per puzzle

    @property
    def solve_rate(self) -> float:
        return self.solved_count / self.total if self.total > 0 else 0.0

    @property
    def avg_guesses_solved(self) -> float:
        return sum(self.guesses_used) / len(self.guesses_used) if self.guesses_used else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def print_summary(self):
        print("=" * 50)
        print("Wordle Benchmark Results")
        print("=" * 50)
        print(f"Puzzles evaluated:    {self.total}")
        print(f"Solved:               {self.solved_count}/{self.total} ({self.solve_rate:.1%})")
        if self.guesses_used:
            print(f"Avg guesses (solved): {self.avg_guesses_solved:.2f}")
        print(f"Avg score:            {self.avg_score:.1f}")
        print("=" * 50)

    def save(self, path: str):
        data = {
            "solved_count": self.solved_count,
            "total": self.total,
            "solve_rate": self.solve_rate,
            "avg_guesses_solved": self.avg_guesses_solved,
            "avg_score": self.avg_score,
            "per_puzzle_scores": self.scores,
            "guesses_used_solved": self.guesses_used,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


class WordleBenchmark:
    """
    Evaluates a fine-tuned model over the full WordleDataset benchmark.

    Loads the model from a path (merged model or LoRA adapter), plays full
    multi-turn Wordle games using WordleEnv, and reports solve rate, avg
    guesses, and score.

    Usage:
        benchmark = WordleBenchmark(model_path='./grpo_wordle_output/merged')
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('results.json')
    """

    def __init__(self, model_path: str, word_set: Optional[set] = None):
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

        self.dataset = WordleDataset()
        self.word_set = word_set or load_dictionary(length=WORDLE_WORD_LENGTH)

        self._system_prompt = (_PROMPTS_DIR / "wordle_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "wordle_user.md").read_text().strip()

    def _generate_guess(self, messages: list, temperature: float = 0.0, max_new_tokens: int = 8) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Fallback for older transformers without Qwen3 thinking support
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
        return "".join(c for c in word if c.isalpha()).upper()[:5]

    def run(
        self,
        num_puzzles: Optional[int] = None,
        max_guesses: int = 6,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Play full multi-turn Wordle games over the dataset.

        Args:
            num_puzzles: number of puzzles to evaluate (None = all ~2300)
            max_guesses: guesses allowed per game
            temperature: 0.0 = greedy decoding (recommended for eval)
            verbose: print progress every 50 puzzles
        """
        n = min(num_puzzles, len(self.dataset)) if num_puzzles else len(self.dataset)
        solved_count = 0
        guesses_used = []
        scores = []

        for idx in range(n):
            item = self.dataset[idx]
            config = self.dataset.get_config(item["puzzle_id"], max_guesses=max_guesses)
            env = WordleEnv(config)
            obs, _ = env.reset()

            user_msg = self._user_prompt_template.format(max_guesses=max_guesses)
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]

            for _ in range(max_guesses):
                guess = self._generate_guess(messages, temperature)
                messages.append({"role": "assistant", "content": guess})
                obs, _, terminated, truncated, _ = env.step(guess)

                if terminated or truncated:
                    break

                remaining = max_guesses - obs["num_guesses"]
                fb = f"Feedback: {obs['feedback']}\n{remaining} attempt(s) remaining. Guess another word."
                messages.append({"role": "user", "content": fb})

            if obs["solved"]:
                solved_count += 1
                guesses_used.append(obs["num_guesses"])
            scores.append(obs["total_points"])

            if verbose and (idx + 1) % 50 == 0:
                rate = solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Solved: {solved_count}/{idx + 1} ({rate:.1%})")

        return BenchmarkResults(
            solved_count=solved_count,
            total=n,
            guesses_used=guesses_used,
            scores=scores,
        )
