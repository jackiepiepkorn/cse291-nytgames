import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..env.connections import ConnectionsEnv
from ..data.dataset import ConnectionsDataset
from .backend import GuessBackend, extract_answer

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class ConnectionsBenchmarkResults:
    fully_solved_count: int         # puzzles where all 4 categories found
    total: int
    categories_found_counts: list   # categories found per puzzle (0–4)
    scores: list                    # total_points per puzzle

    @property
    def solve_rate(self) -> float:
        return self.fully_solved_count / self.total if self.total > 0 else 0.0

    @property
    def avg_categories_found(self) -> float:
        return sum(self.categories_found_counts) / len(self.categories_found_counts) if self.categories_found_counts else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def print_summary(self):
        print("=" * 50)
        print("Connections Benchmark Results")
        print("=" * 50)
        print(f"Puzzles evaluated:    {self.total}")
        print(f"Fully solved:         {self.fully_solved_count}/{self.total} ({self.solve_rate:.1%})")
        print(f"Avg categories found: {self.avg_categories_found:.2f}/4")
        print(f"Avg score:            {self.avg_score:.1f}")
        print("=" * 50)

    def save(self, path: str):
        data = {
            "fully_solved_count": self.fully_solved_count,
            "total": self.total,
            "solve_rate": self.solve_rate,
            "avg_categories_found": self.avg_categories_found,
            "avg_score": self.avg_score,
            "per_puzzle_scores": self.scores,
            "per_puzzle_categories_found": self.categories_found_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


class ConnectionsBenchmark:
    """
    Evaluates a model over the ConnectionsDataset benchmark.

    The model must output structured responses with <reasoning> and <guess> tags
    as specified in connections_system.md. The <guess> tag should contain exactly
    4 comma-separated words.

    Usage:
        benchmark = ConnectionsBenchmark(backend)
        results = benchmark.run(num_puzzles=100)
        results.print_summary()
        results.save('connections_results.json')
    """

    def __init__(self, backend: GuessBackend):
        self.backend = backend
        self.dataset = ConnectionsDataset()
        self._system_prompt = (_PROMPTS_DIR / "connections_system.md").read_text().strip()
        self._user_prompt_template = (_PROMPTS_DIR / "connections_user.md").read_text().strip()

    def _generate_response(self, messages: list, temperature: float = 0.0, max_new_tokens: int = 200) -> str:
        raw = self.backend.generate_text(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        if self.backend.use_thinking_format:
            raw = extract_answer(raw)
        return raw

    @staticmethod
    def _parse_guess(response: str) -> str:
        """Extract the <guess>...</guess> content from the model response."""
        match = re.search(r"<guess>(.*?)</guess>", response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return raw response so env can handle it (will give 0 reward)
        return response

    def run(
        self,
        num_puzzles: Optional[int] = None,
        max_mistakes: int = 4,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> ConnectionsBenchmarkResults:
        """
        Play full multi-turn Connections games over the dataset.

        Args:
            num_puzzles: number of puzzles to evaluate (None = all)
            max_mistakes: wrong guesses allowed per game
            temperature: 0.0 = greedy decoding (recommended for eval)
            verbose: print progress every 50 puzzles
        """
        n = min(num_puzzles, len(self.dataset)) if num_puzzles else len(self.dataset)
        fully_solved_count = 0
        categories_found_counts = []
        scores = []

        for idx in range(n):
            item = self.dataset[idx]
            config = self.dataset.get_config(item["puzzle_id"])
            # Override max_mistakes if specified
            config.max_mistakes = max_mistakes
            env = ConnectionsEnv(config)
            obs, _ = env.reset()

            words_display = ", ".join(obs["remaining_words"])
            user_msg = self._user_prompt_template.format(words=words_display)
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]

            terminated = False
            truncated = False
            # Max turns = 4 categories + max_mistakes wrong guesses
            max_turns = 4 + max_mistakes
            for _ in range(max_turns):
                response = self._generate_response(messages, temperature)
                messages.append({"role": "assistant", "content": response})

                action = self._parse_guess(response)
                obs, reward, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break

                # Feedback message (mirror LLMHandler._connections_feedback)
                feedback = obs.get("feedback", "")
                mistakes_left = obs.get("mistakes_remaining", "?")

                if reward >= 1:
                    solved = obs.get("solved_categories", {})
                    last_category = list(solved.keys())[-1]
                    fb = (
                        f"Correct! Category '{last_category}' found. +{int(reward)} point(s).\n"
                        f"Solved so far: {', '.join(solved.keys())}"
                    )
                else:
                    fb = f"{feedback} Mistakes remaining: {mistakes_left}"

                remaining = obs.get("remaining_words", [])
                if remaining:
                    fb += f"\nRemaining words: {', '.join(remaining)}.\nGuess 4 words separated by commas."
                else:
                    fb += "\nPuzzle complete!"
                messages.append({"role": "user", "content": fb})

            if terminated:
                fully_solved_count += 1

            categories_found_counts.append(len(obs.get("solved_categories", {})))
            scores.append(env.total_points)

            if verbose and (idx + 1) % 50 == 0:
                rate = fully_solved_count / (idx + 1)
                print(f"[{idx + 1}/{n}] Fully solved: {fully_solved_count}/{idx + 1} ({rate:.1%})")

        return ConnectionsBenchmarkResults(
            fully_solved_count=fully_solved_count,
            total=n,
            categories_found_counts=categories_found_counts,
            scores=scores,
        )
