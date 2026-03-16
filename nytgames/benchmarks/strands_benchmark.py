import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..env.strands import StrandsEnv
from ..data.dataset import StrandsDataset, load_dictionary
from .backend import GuessBackend, extract_answer

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class StrandsBenchmarkResults:
    fully_solved_count: int            # puzzles where all theme words found
    total: int
    theme_words_found_counts: list     # theme words found per puzzle
    total_theme_words_counts: list     # total theme words per puzzle
    scores: list                       # total reward per puzzle
    spanagram_found_per_puzzle: list   # bool per puzzle: was spanagram found?

    @property
    def spanagram_find_rate(self) -> float:
        return sum(self.spanagram_found_per_puzzle) / len(self.spanagram_found_per_puzzle) \
            if self.spanagram_found_per_puzzle else 0.0

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
        print(f"Spanagram find rate:    {self.spanagram_find_rate:.1%}")
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
            "spanagram_find_rate": self.spanagram_find_rate,
            "per_puzzle_spanagram_found": self.spanagram_found_per_puzzle,
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
        benchmark = StrandsBenchmark(backend)
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('strands_results.json')
    """

    def __init__(self, backend: GuessBackend):
        self.backend = backend
        self.dataset = StrandsDataset()
        self._system_prompt = (_PROMPTS_DIR / "strands_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "strands_user.md").read_text().strip()

    def _generate_guess(self, messages: list, temperature: float = 0.0, max_new_tokens: int = 10) -> str:
        raw = self.backend.generate_text(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        if self.backend.use_thinking_format:
            raw = extract_answer(raw)
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
        spanagram_found_per_puzzle = []

        dictionary = load_dictionary()

        for idx in range(n):
            item = self.dataset[idx]
            config = self.dataset.get_config(item["puzzle_id"])
            config.dictionary = dictionary  # enables hint system
            env = StrandsEnv(config)
            obs, _ = env.reset()

            max_guesses = config.max_guesses  # 3 × num_theme_words by default
            num_theme_words = len(config.theme_words)
            already_guessed = set()

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
            spanagram_found = False
            for _ in range(max_guesses):
                # Retry up to 10 times to get a non-empty, non-repeated word
                guess = None
                for _ in range(10):
                    candidate = self._generate_guess(messages, temperature)
                    if candidate and candidate not in already_guessed:
                        guess = candidate
                        break
                if guess is None:
                    guess = candidate or "SKIP"
                already_guessed.add(guess)

                messages.append({"role": "assistant", "content": guess})
                obs, reward, terminated, truncated, _ = env.step(guess)

                if reward == 5:
                    spanagram_found = True

                if terminated or truncated:
                    break

                # Feedback message
                feedback = obs.get("feedback", "")
                if reward == 5:
                    fb = f"'{guess}': {feedback} Spanagram! +5 points."
                elif reward == 1:
                    fb = f"'{guess}': {feedback} +1 point."
                else:
                    fb = f"'{guess}': {feedback}"
                found = ", ".join(obs["theme_words_guessed"]) if obs["theme_words_guessed"] else "none"
                fb += (
                    f"\nFound so far: {found}."
                    f"\n\nUpdated board:\n{obs['board_str']}"
                    f"\nPlease guess another board word."
                )
                messages.append({"role": "user", "content": fb})

            if terminated:
                fully_solved_count += 1

            theme_words_found_counts.append(len(obs["theme_words_guessed"]))
            total_theme_words_counts.append(num_theme_words)
            total_reward = sum(r for _, r, _ in env.info.get("history", []))
            scores.append(total_reward)
            spanagram_found_per_puzzle.append(spanagram_found)

            if verbose and (idx + 1) % 50 == 0:
                rate = fully_solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Fully solved: {fully_solved_count}/{idx + 1} ({rate:.1%})")

        return StrandsBenchmarkResults(
            fully_solved_count=fully_solved_count,
            total=n,
            theme_words_found_counts=theme_words_found_counts,
            total_theme_words_counts=total_theme_words_counts,
            scores=scores,
            spanagram_found_per_puzzle=spanagram_found_per_puzzle,
        )
