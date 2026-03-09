from dataclasses import dataclass
from .base import NYTGameEnv

WORDLE_MAX_GUESSES = 6
WORDLE_WORD_LENGTH = 5

@dataclass
class WordleConfig:
    """
    Configuration for a Wordle puzzle.

    Attributes:
        target_word: The 5-letter word the agent must guess.
        word_set: Set of valid 5-letter words that can be guessed (must include target_word).
        max_guesses: Episode length; game truncates after this many guesses.
    """
    target_word: str
    word_set: set[str]
    max_guesses: int = WORDLE_MAX_GUESSES

    def __post_init__(self):
        self.target_word = self.target_word.upper()
        self.word_set = {word.upper() for word in self.word_set}
        self.validate()

    def validate(self):
        assert len(self.target_word) == WORDLE_WORD_LENGTH, f"Target word must be {WORDLE_WORD_LENGTH} letters long"
        assert self.target_word in self.word_set, "Target word must be in the word set"
        assert all(len(word) == WORDLE_WORD_LENGTH for word in self.word_set), f"All words in the word set must be {WORDLE_WORD_LENGTH} letters long"
        assert self.max_guesses > 0, "Max guesses must be a positive integer"

#Tile states
CORRECT = "correct"  # green color, right color and right position
PRESENT = "present"  # yellow color, right color but wrong position
ABSENT = "absent"    # gray color, letter not in word at all

def _score_guess(guess: str, target: str) -> list[str]:
    """
    Return a per letter result list for a single guess against target
    i.e. list of CORRECT / PRESENT / ABSENT, one entry per letter
    """
    result = [ABSENT] * WORDLE_WORD_LENGTH
    target_remaining = list(target)  # to keep track of which letters in target have been matched

    for i, (g,t) in enumerate(zip(guess, target)):
        if g == t:
            result[i] = CORRECT
            target_remaining[i] = None  # mark this letter as matched

    for i, g in enumerate(guess):
        if result[i] == CORRECT:
            continue  # already scored as correct
        if g in target_remaining:
            result[i] = PRESENT
            target_remaining[target_remaining.index(g)] = None  # mark this letter as matched

    return result

def _format_feedback(guess: str, tile_results: list[str]) -> str:
    """
    Return human readable string showing tile results for guess
    """
    symbols = {
        CORRECT: "🟩",
        PRESENT: "🟨",
        ABSENT: "⬜"
    }
    tiles = "".join(symbols[tile] for tile in tile_results)
    return f"{tiles}  {guess}"


class WordleEnv(NYTGameEnv):
    """
    Gymnasium-compatible environment for the NYT Wordle.

    Observation (dict):
        num_guesses:    number of guesses made so far
        guesses:        list of all guesses made so far (only valid format ones)
        feedback:       human-readable result of the last guess, or None on reset
        keyboard:       dict mapping letters to their current known status (CORRECT, PRESENT, ABSENT, or None if not guessed yet)
        solved:         boolean indicating if the puzzle has been solved

    Action:
        A string representing a 5-letter guess (case-insensitive).

    Reward:
        0: invalid guess (wrong length, not in word list, or already guessed)
        1: valid guess, not the answer
        10: correct answer on any attempt
        Bonus for guessing early: +2 per remaining guess after the winning one

    Termination: target word guessed correctly
    Truncation: max_guesses reached without guessing target word
    """

    def __init__(self, config: WordleConfig, render_mode: str="human"):
        super().__init__(config)
        self.render_mode = render_mode
        self._reset_state()

    # GYMNASIUM INTERFACE METHODS
    def reset(self) -> tuple[dict, dict]:
        self._reset_state()
        return self._get_obs(), self.info  # Return initial observation and info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        action = action.strip().upper()

        # Reject repeated guesses without consuming a turn (like real Wordle)
        if action in self._guessed_words:
            feedback = f"Invalid guess: '{action}' was already guessed."
            self.info['history'].append((action, 0, feedback))
            return self._get_obs(), 0, self._is_terminated(), self._is_truncated(), self.info

        self.num_guesses += 1
        valid, feedback, tile_results = self._process_guess(action)

        if valid:
            self._guessed_words.add(action)
            self.guesses.append((action, tile_results))
            self._update_keyboard(action, tile_results)
            if action == self.config.target_word:
                self.solved = True
            reward = self._get_reward(action)
        else:
            reward = 0

        self.total_points += reward
        self.info['history'].append((action, reward, feedback))

        return self._get_obs(), reward, self._is_terminated(), self._is_truncated(), self.info

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n{'='*30}")
        print(f"Wordle | Guess {self.num_guesses}/{self.config.max_guesses}")
        for word, tiles in self.guesses:
            print(_format_feedback(word, tiles))
        # show blank rows
        for _ in range(self.config.max_guesses - len(self.guesses)):
            print("⬜⬜⬜⬜⬜")
        print(f"\nTotal points: {self.total_points} ")
        print(f"Keyboard: {self._keyboard_display()}")

    def close(self):
        pass

    def _get_obs(self) -> dict:
        obs = {
            'num_guesses': self.num_guesses,
            'total_points': self.total_points,
            'guesses': list(self.guesses),  # list of (guess, tile_results)
            'feedback': self.info['history'][-1][2] if self.info['history'] else None,  # feedback from last guess
            'keyboard': dict(self.keyboard),
            'solved': self.solved
        }
        return obs

    def _get_reward(self, action: str) -> float:
        if action == self.config.target_word:
            # Base reward for solving the puzzle
            reward = 10
            # Bonus for guessing early
            reward += 2 * (self.config.max_guesses - self.num_guesses)
            return reward
        else:
            return 1  # valid guess but not correct answer

    def _is_terminated(self) -> bool:
        return self.solved

    def _is_truncated(self) -> bool:
        return self.num_guesses >= self.config.max_guesses and not self.solved

    # INTERNAL HELPER METHODS
    def _reset_state(self):
        self.num_guesses = 0
        self.total_points = 0
        self.guesses = []  # list of (guess, tile_results)
        self._guessed_words = set()  # track unique guesses for repeat detection
        self.keyboard = {chr(c): None for c in range(ord('A'), ord('Z')+1)}  # letter -> CORRECT/PRESENT/ABSENT/None
        self.solved = False
        self.info = {'history': []}  # (guess, reward, feedback)

    def _process_guess(self, guess: str) -> tuple[bool, str, list[str] | None]:
        if len(guess) != WORDLE_WORD_LENGTH:
            return False, f"Invalid guess: '{guess}' is not {WORDLE_WORD_LENGTH} letters long.", None
        if guess not in self.config.word_set:
            return False, f"Invalid guess: '{guess}' is not in the word list.", None

        tile_results = _score_guess(guess, self.config.target_word)
        feedback = _format_feedback(guess, tile_results)
        if guess == self.config.target_word:
            feedback += " - Correct!"

        return True, feedback, tile_results

    def _update_keyboard(self, guess: str, tile_results: list[str]):
        priority = {None: 0, ABSENT: 1, PRESENT: 2, CORRECT: 3}
        for letter, result in zip(guess, tile_results):
            current = self.keyboard[letter]
            if priority[result] > priority[current]:
                self.keyboard[letter] = result

    def _keyboard_display(self) -> str:
        symbols = {CORRECT: "🟩", PRESENT: "🟨", ABSENT: "⬜", None: " "}
        return " ".join(f"{letter}{symbols[status]}"for letter, status in self.keyboard.items())

if __name__ == "__main__":
    # Quick manual test with a tiny word list
    from nytgames import load_dictionary

    word_set = load_dictionary(length=WORDLE_WORD_LENGTH)

    config = WordleConfig(
        target_word="CRANE",
        word_set=word_set, #{"CRANE", "SLATE", "AUDIO", "AROSE", "STARE", "TRAIN", "PLAIN", "BRAIN"},
        max_guesses=6,
    )

    truncated = False
    terminated = False

    env = WordleEnv(config)
    env.render()

    while not (truncated or terminated):
        guess = input("\nEnter your 5-letter guess: ").strip().upper()
        obs, reward, terminated, truncated, info = env.step(guess)
        print(f"\n{obs['feedback']}")
        print(f"Total points: {reward} |  Guesses: {obs['num_guesses']}")
        env.render()

    if obs["solved"]:
        print(f"\n:) Solved in {obs['num_guesses']} guess(es)! Final score: {reward}")
    else:
        print(f"\n:( Out of guesses. The word was: {config.target_word}")



