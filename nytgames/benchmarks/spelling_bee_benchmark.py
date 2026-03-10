import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ..env.spellingbee import SpellingBeeEnv
from ..data.dataset import SpellingBeeDataset

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class SpellingBeeBenchmarkResults:
    fully_solved_count: int   # puzzles where ALL words found
    total: int
    words_found_counts: list  # words found per puzzle
    total_words_counts: list  # total words available per puzzle
    scores: list              # total_points per puzzle

    @property
    def solve_rate(self) -> float:
        return self.fully_solved_count / self.total if self.total > 0 else 0.0

    @property
    def avg_words_found_pct(self) -> float:
        if not self.words_found_counts:
            return 0.0
        fracs = [f / t for f, t in zip(self.words_found_counts, self.total_words_counts) if t > 0]
        return sum(fracs) / len(fracs) if fracs else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def print_summary(self):
        print("=" * 50)
        print("Spelling Bee Benchmark Results")
        print("=" * 50)
        print(f"Puzzles evaluated:    {self.total}")
        print(f"Fully solved:         {self.fully_solved_count}/{self.total} ({self.solve_rate:.1%})")
        print(f"Avg words found:      {self.avg_words_found_pct:.1%}")
        print(f"Avg score:            {self.avg_score:.1f}")
        print("=" * 50)

    def save(self, path: str):
        data = {
            "fully_solved_count": self.fully_solved_count,
            "total": self.total,
            "solve_rate": self.solve_rate,
            "avg_words_found_pct": self.avg_words_found_pct,
            "avg_score": self.avg_score,
            "per_puzzle_scores": self.scores,
            "per_puzzle_words_found": self.words_found_counts,
            "per_puzzle_total_words": self.total_words_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


class SpellingBeeBenchmark:
    """
    Evaluates a model over the SpellingBeeDataset benchmark.

    Usage:
        benchmark = SpellingBeeBenchmark(model_path='./model')
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('spelling_bee_results.json')
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

        self.dataset = SpellingBeeDataset()
        self._system_prompt = (_PROMPTS_DIR / "spelling_bee_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "spelling_bee_user.md").read_text().strip()

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
        max_guesses: int = 10,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> SpellingBeeBenchmarkResults:
        """
        Play full multi-turn Spelling Bee games over the dataset.

        Args:
            num_puzzles: number of puzzles to evaluate (None = all)
            max_guesses: guesses allowed per game
            temperature: 0.0 = greedy decoding (recommended for eval)
            verbose: print progress every 50 puzzles
        """
        n = min(num_puzzles, len(self.dataset)) if num_puzzles else len(self.dataset)
        fully_solved_count = 0
        words_found_counts = []
        total_words_counts = []
        scores = []

        for idx in range(n):
            item = self.dataset[idx]
            config = self.dataset.get_config(item["puzzle_id"], max_guesses=max_guesses)
            env = SpellingBeeEnv(config)
            obs, _ = env.reset()

            user_msg = self._user_prompt_template.format(
                letters=", ".join(sorted(config.letter_set)),
                center=config.center_letter,
                max_guesses=max_guesses,
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

                # Feedback message
                feedback = obs.get("feedback", "")
                if reward > 0:
                    fb = (
                        f"'{guess}': {feedback} You earned {reward} point(s). "
                        f"Running total: {obs['total_points']}."
                    )
                else:
                    fb = f"'{guess}': {feedback} No points earned."

                found = ", ".join(obs["valid_words_guessed"]) if obs["valid_words_guessed"] else "none"
                fb += (
                    f"\nGuesses used: {obs['num_guesses']}. "
                    f"Words found so far: {found}.\n"
                    f"Please guess another word."
                )
                messages.append({"role": "user", "content": fb})

            if terminated:
                fully_solved_count += 1

            words_found_counts.append(len(obs["valid_words_guessed"]))
            total_words_counts.append(len(config.word_set))
            scores.append(obs["total_points"])

            if verbose and (idx + 1) % 50 == 0:
                rate = fully_solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Fully solved: {fully_solved_count}/{idx + 1} ({rate:.1%})")

        return SpellingBeeBenchmarkResults(
            fully_solved_count=fully_solved_count,
            total=n,
            words_found_counts=words_found_counts,
            total_words_counts=total_words_counts,
            scores=scores,
        )
