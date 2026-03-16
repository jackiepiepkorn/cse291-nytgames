import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..env.wordle import WordleEnv, WordleConfig, WORDLE_WORD_LENGTH, _score_guess
from ..data.dataset import WordleDataset, load_dictionary
from .backend import GuessBackend, extract_answer

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class WordleBenchmarkResults:
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
    Evaluates a model over the full WordleDataset benchmark.

    Accepts a GuessBackend (HFBackend or CloudBackend) for generation.
    Plays full multi-turn Wordle games using WordleEnv and reports solve rate,
    avg guesses, and score.

    Usage:
        from nytgames.benchmarks.backend import HFBackend
        backend = HFBackend(model, tokenizer)
        benchmark = WordleBenchmark(backend)
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('results.json')
    """

    def __init__(self, backend: GuessBackend, word_set: Optional[set] = None):
        self.backend = backend
        self.dataset = WordleDataset()
        self.word_set = word_set or load_dictionary(length=WORDLE_WORD_LENGTH)
        self._system_prompt = (_PROMPTS_DIR / "wordle_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "wordle_user.md").read_text().strip()

    def _generate_guess(self, messages: list, temperature: float = 0.0, max_new_tokens: int = 8) -> str:
        raw = self.backend.generate_text(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        if self.backend.use_thinking_format:
            raw = extract_answer(raw)
        word = raw.split()[0] if raw.split() else raw
        return "".join(c for c in word if c.isalpha()).upper()[:5]

    def run(
        self,
        num_puzzles: Optional[int] = None,
        max_guesses: int = 6,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> WordleBenchmarkResults:
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

            # Constraint state
            green = {}        # position → letter
            yellow = {}       # position → set of letters (right letter, wrong spot)
            gray = set()      # letters confirmed absent
            previous_guesses = set()

            for _ in range(max_guesses):
                # Retry up to 10 times to get a valid, non-repeated dictionary word
                guess = None
                best_fallback = None
                for _ in range(10):
                    candidate = self._generate_guess(messages, temperature)
                    if len(candidate) == 5 and candidate in self.word_set and candidate not in previous_guesses:
                        guess = candidate
                        break
                    if best_fallback is None and candidate not in previous_guesses:
                        best_fallback = candidate
                if guess is None:
                    guess = best_fallback or candidate

                previous_guesses.add(guess)
                messages.append({"role": "assistant", "content": guess})
                obs, _, terminated, truncated, _ = env.step(guess)

                if terminated or truncated:
                    break

                # Update constraints from tile results
                tiles = _score_guess(guess, config.target_word)
                for pos, (letter, tile) in enumerate(zip(guess, tiles)):
                    if tile == "correct":
                        green[pos] = letter
                    elif tile == "present":
                        yellow.setdefault(pos, set()).add(letter)
                    else:
                        gray.add(letter)

                # Build cumulative constraint summary
                hints = []
                if green:
                    hints.append("Confirmed: " + ", ".join(
                        f"{l} at position {p + 1}" for p, l in sorted(green.items())
                    ))
                present = {l for s in yellow.values() for l in s} - set(green.values())
                if present:
                    hints.append(f"In word but position unknown: {', '.join(sorted(present))}")
                truly_gray = gray - set(green.values()) - present
                if truly_gray:
                    hints.append(f"Not in word: {', '.join(sorted(truly_gray))}")

                remaining = max_guesses - obs["num_guesses"]
                fb = f"Feedback: {obs['feedback']}\nYou have {remaining} attempt(s) remaining."
                if hints:
                    fb += "\n\nWhat we know:\n" + "\n".join(hints)
                fb += f"\nPrevious guesses: {', '.join(sorted(previous_guesses))}. Do NOT repeat any of these."
                fb += "\nPlease guess another 5-letter word."
                messages.append({"role": "user", "content": fb})

            if obs["solved"]:
                solved_count += 1
                guesses_used.append(obs["num_guesses"])
            scores.append(obs["total_points"])

            if verbose and (idx + 1) % 50 == 0:
                rate = solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Solved: {solved_count}/{idx + 1} ({rate:.1%})")

        return WordleBenchmarkResults(
            solved_count=solved_count,
            total=n,
            guesses_used=guesses_used,
            scores=scores,
        )
