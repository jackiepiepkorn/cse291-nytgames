from dataclasses import dataclass
from .base import NYTGameEnv

@dataclass
class SpellingBeeConfig:
    """
    Configuration for a Spelling Bee puzzle.

    Attributes:
        center_letter: The required letter that must appear in every valid word.
        letter_set:    All 7 allowed letters (includes center_letter).
        word_set:      The set of valid solution words for this puzzle.
        max_guesses:   Episode length; game truncates after this many guesses.
    """
    center_letter: str
    letter_set: set[str]
    word_set: set[str]
    max_guesses: int = 10

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert self.center_letter in self.letter_set, "Center letter must be in the letter set"
        assert all(letter in self.letter_set for word in self.word_set for letter in word), "All words must be formed from the letter set"
        for word in self.word_set:
            assert any(letter == self.center_letter for letter in word), "All words must contain the center letter"
        assert self.max_guesses > 0, "Max guesses must be a positive integer"



class SpellingBeeEnv(NYTGameEnv):
    """
    Gymnasium-compatible environment for the NYT Spelling Bee.

    Observation (dict):
        num_guesses:        number of guesses made so far
        total_points:       cumulative score
        valid_words_guessed: list of correctly found words (shown on board, like real game)
        feedback:           human-readable result of the last guess, or None on reset
        (full attempt history including invalid guesses is in info['history'])

    Action:
        A string word guess (case-insensitive).

    Reward:
        0: invalid guess or word shorter than 4 letters
        1: valid 4-letter word
        n: valid word of length n (n > 4), +7 bonus if pangram

    Termination: all words in word_set have been found.
    Truncation:  max_guesses reached.
    """
    def __init__(self, config: SpellingBeeConfig, render_mode: str="human"):
        super().__init__(config)
        self.render_mode = render_mode

        self.num_guesses = 0
        self.total_points = 0
        self.valid_words_guessed = set()
        self.info= {'history': []} # (guess, reward, feedback)

    def reset(self) -> tuple[dict, dict]:
        self.num_guesses = 0
        self.total_points = 0
        self.valid_words_guessed = set()
        self.info = {'history': []}  # (guess, reward, feedback)
        return self._get_obs(), self.info  # Return initial observation and info

    def _get_obs(self) -> dict:
        obs = {
            'num_guesses': self.num_guesses,
            'total_points': self.total_points,
            'valid_words_guessed': list(self.valid_words_guessed),
            'feedback': self.info['history'][-1][2] if self.info['history'] else None
        }
        return obs

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        action = action.strip().upper()

        self.num_guesses += 1
        valid, feedback = self._word_is_valid(action)

        if not valid:
            reward = 0
        else:
            self.valid_words_guessed.add(action)
            reward = self._get_reward(action)

        self.total_points += reward
        self.info['history'].append((action, reward, feedback))
        return self._get_obs(), reward, self._is_terminated(), self._is_truncated(), self.info

    def render(self):
        if self.render_mode == "human":
            print(f"Center letter: {self.config.center_letter} | All letters: {self.config.letter_set}")
            print(f"Total points: {self.total_points} | You have found {len(self.valid_words_guessed)} words: {self.valid_words_guessed}")

    def close(self):
        pass  # No cleanup needed for this environment

    def _get_reward(self, word) -> float:
        """4-letter words score 1; longer words score their length; pangrams get +7."""
        reward = 0
        if len(word) < 4:
            return 0
        elif len(word) == 4:
            return 1
        else:
            reward += len(word)
            if self._is_pangram(word):
                reward += 7
            return reward

    def _is_pangram(self, word) -> bool:
        return set(word) >= set(self.config.letter_set)

    def _word_is_valid(self, word) -> tuple[bool, str]:
        """
        Checks if word is a valid guess.

        :returns: (valid, feedback) where feedback is a human-readable explanation
        """
        invalid_letters = set(word) - self.config.letter_set
        if len(word) < 4:
            return False, "Too short"
        elif self.config.center_letter not in word:
            return False, "Missing center letter"
        elif invalid_letters:
            return False, f"Invalid letters: {', '.join(sorted(invalid_letters))}"
        elif word in self.valid_words_guessed:
            return False, "Already found!"
        elif word not in self.config.word_set:
            return False, "Not in word list"

        msg = "Nice, found new word!"
        if self._is_pangram(word):
            msg += " Panagram!"
        return True, msg


    def _is_terminated(self) -> bool:
        """True when all words in word_set have been found."""
        return len(self.config.word_set) == len(self.valid_words_guessed)

    def _is_truncated(self) -> bool:
        """True when max_guesses have been used."""
        return self.num_guesses >= self.config.max_guesses


if __name__ == "__main__":
    from nytgames.data import SpellingBeeDataset

    # Option A: load from dataset by puzzle id
    dataset = SpellingBeeDataset(max_guesses=10)
    config = dataset.get_config(puzzle_id=7)

    # Option B: manually specify a config
    # config = SpellingBeeConfig(
    #     center_letter="N",
    #     letter_set={"T", "A", "F", "M", "O", "R", "N"},
    #     word_set={"FRONTMAN", "FONT", "AFFRONT", "MAROON", "TORN"},
    #     max_guesses=10,
    # )

    # Init variables to track game status
    truncated = False   # True if max guesses are used
    terminated = False  # True if all valid words are guessed

    env = SpellingBeeEnv(config)
    while not (truncated or terminated):
        print()
        env.render()

        guess = input("Enter your guess: ").strip().upper()
        obs, reward, terminated, truncated, info = env.step(guess)
        print(f"\n{obs['feedback']}\nReward: {reward}, Total Points: {obs['total_points']}, Num Guesses: {obs['num_guesses']}")
        if truncated or terminated:
            print("Game Over!")
            break

