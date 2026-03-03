from dataclasses import dataclass
from typing import Dict, List
import random
from .base import NYTGameEnv

CONNECTIONS_MAX_MISTAKES = 4
CATEGORIES_COUNT = 4
WORDS_PER_CATEGORY = 4

@dataclass
class ConnectionsConfig:
    """
    Configuration for a Connections puzzle.

    Attributes:
        categories: A dictionary mapping the category name to a list of 4 words.
                    e.g., {"FRUITS": ["APPLE", "BANANA", "CHERRY", "DATE"]}
        max_mistakes: Number of incorrect guesses allowed before truncation.
    """
    categories: Dict[str, List[str]]
    max_mistakes: int = CONNECTIONS_MAX_MISTAKES

    def __post_init__(self):
        # Normalize to uppercase
        self.categories = {
            k.upper(): [w.upper() for w in v]
            for k, v in self.categories.items()
        }
        self.validate()

    def validate(self):
        assert len(self.categories) == CATEGORIES_COUNT, f"Must have exactly {CATEGORIES_COUNT} categories"
        all_words = []
        for cat, words in self.categories.items():
            assert len(words) == WORDS_PER_CATEGORY, f"Category '{cat}' must have exactly {WORDS_PER_CATEGORY} words"
            all_words.extend(words)
        
        unique_words = set(all_words)
        assert len(unique_words) == CATEGORIES_COUNT * WORDS_PER_CATEGORY, "All words in the puzzle must be unique"


class ConnectionsEnv(NYTGameEnv):
    """
    Gymnasium-compatible environment for the NYT Connections game.

    Observation (dict):
        remaining_words:    list of words still on the board
        solved_categories:  dict of discovered categories and their words
        mistakes_remaining: number of incorrect guesses left
        feedback:           human-readable result of the last guess, or None on reset

    Action:
        A list of 4 string words, or a single comma-separated string (case-insensitive).

    Reward:
        0: invalid guess (wrong length, not on board, already guessed) or incorrect guess
        1: correct category found
        10: bonus for completing the puzzle

    Termination: all categories have been found.
    Truncation: max_mistakes reached.
    """
    def __init__(self, config: ConnectionsConfig, render_mode: str = "human"):
        super().__init__(config)
        self.render_mode = render_mode
        self._reset_state()

    def reset(self) -> tuple[dict, dict]:
        self._reset_state()
        return self._get_obs(), self.info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        # Handle string input (e.g., "word1, word2, word3, word4") or list input
        if isinstance(action, str):
            guess = [w.strip().upper() for w in action.split(',')]
        else:
            guess = [w.strip().upper() for w in action]

        # 1. Input Validation
        if len(guess) != WORDS_PER_CATEGORY:
            feedback = f"Invalid guess: You must select exactly {WORDS_PER_CATEGORY} words."
            self.info['history'].append((guess, 0.0, feedback))
            return self._get_obs(), 0.0, self._is_terminated(), self._is_truncated(), self.info
        
        if len(set(guess)) != WORDS_PER_CATEGORY:
            feedback = "Invalid guess: Words must be unique."
            self.info['history'].append((guess, 0.0, feedback))
            return self._get_obs(), 0.0, self._is_terminated(), self._is_truncated(), self.info

        if not all(w in self.remaining_words for w in guess):
            feedback = "Invalid guess: One or more words are not on the board."
            self.info['history'].append((guess, 0.0, feedback))
            return self._get_obs(), 0.0, self._is_terminated(), self._is_truncated(), self.info

        guess_set = frozenset(guess)

        # 2. Check for previously guessed combinations
        if guess_set in self.past_guesses:
            feedback = "Already guessed!"
            self.info['history'].append((guess, 0.0, feedback))
            return self._get_obs(), 0.0, self._is_terminated(), self._is_truncated(), self.info
        
        self.past_guesses.add(guess_set)

        # 3. Evaluate the guess
        is_correct = False
        is_one_away = False
        matched_category = None

        for cat, words in self.config.categories.items():
            if cat in self.solved_categories:
                continue
            cat_set = set(words)
            intersect = guess_set.intersection(cat_set)
            
            if len(intersect) == 4:
                is_correct = True
                matched_category = cat
                break
            elif len(intersect) == 3:
                is_one_away = True

        # 4. State update and reward calculation
        reward = 0.0
        if is_correct:
            feedback = f"Correct! '{matched_category}'"
            self.solved_categories[matched_category] = self.config.categories[matched_category]
            for w in guess:
                self.remaining_words.remove(w)
            
            reward = self._get_reward(matched_category)
        else:
            self.mistakes_made += 1
            feedback = "One away!" if is_one_away else "Incorrect."

        self.total_points += reward
        self.info['history'].append((guess, reward, feedback))

        return self._get_obs(), reward, self._is_terminated(), self._is_truncated(), self.info

    def render(self):
        if self.render_mode != "human":
            return
        
        print(f"\n{'='*40}")
        print("CONNECTIONS")
        print(f"Mistakes remaining: {'⏺ ' * (self.config.max_mistakes - self.mistakes_made)}")
        print("-" * 40)
        
        # Print solved categories
        for cat, words in self.solved_categories.items():
            print(f"🟩 {cat}: {', '.join(words)}")
        
        if self.solved_categories:
            print("-" * 40)

        # Print remaining words in a grid
        for i in range(0, len(self.remaining_words), 4):
            row = self.remaining_words[i:i+4]
            # Format row neatly
            print(" | ".join(f"{w:^12}" for w in row))
        print(f"{'='*40}\n")

    def close(self):
        pass

    def _get_obs(self) -> dict:
        return {
            'remaining_words': list(self.remaining_words),
            'solved_categories': dict(self.solved_categories),
            'mistakes_remaining': self.config.max_mistakes - self.mistakes_made,
            'feedback': self.info['history'][-1][2] if self.info['history'] else None
        }

    def _get_reward(self, category: str) -> float:
        """+1 for finding a category, +10 bonus if it's the final one."""
        reward = 1.0
        # If finding this category results in termination (all 4 found)
        if len(self.solved_categories) == CATEGORIES_COUNT:
            reward += 10.0
        return reward

    def _is_terminated(self) -> bool:
        """True when all categories have been found."""
        return len(self.solved_categories) == CATEGORIES_COUNT

    def _is_truncated(self) -> bool:
        """True when max mistakes have been made."""
        return self.mistakes_made >= self.config.max_mistakes

    def _reset_state(self):
        self.mistakes_made = 0
        self.total_points = 0
        self.solved_categories = {}
        self.past_guesses = set()
        
        # Flatten and shuffle remaining words
        self.remaining_words = []
        for words in self.config.categories.values():
            self.remaining_words.extend(words)
        random.shuffle(self.remaining_words)
        
        self.info = {'history': []}


if __name__ == "__main__":
    # Quick manual test
    config = ConnectionsConfig(
        categories={
            "WEB BROWSERS": ["EDGE", "CHROME", "OPERA", "SAFARI"],
            "SYNONYMS FOR 'FAST'": ["QUICK", "RAPID", "SWIFT", "BRISK"],
            "TYPES OF PASTA": ["PENNE", "ZITI", "RIGATONI", "FARFALLE"],
            "COMPANIES WITH AN 'X'": ["XEROX", "EXXON", "ROLEX", "CLOROX"]
        }
    )

    truncated = False
    terminated = False

    env = ConnectionsEnv(config)
    env.render()

    while not (truncated or terminated):
        guess_str = input("Enter 4 words separated by commas: ").strip()
        obs, reward, terminated, truncated, info = env.step(guess_str)
        
        print(f"\n>>> {obs['feedback']}")
        env.render()

    if terminated:
        print(f"Game Over: You won! Total points: {env.total_points}")
    else:
        print("Game Over: Out of mistakes!")
        print("The remaining categories were:")
        for cat, words in config.categories.items():
            if cat not in env.solved_categories:
                print(f"- {cat}: {', '.join(words)}")