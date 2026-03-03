from dataclasses import dataclass
from .base import NYTGameEnv

@dataclass
class SpellingBeeConfig:
    center_letter: str
    letter_list: set[str]
    word_list: set[str]
    max_guesses: int = 10

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert self.center_letter in self.letter_list, "Center letter must be in the letter list"
        assert all(letter in self.letter_list for word in self.word_list for letter in word), "All words must be formed from the letter list"
        for word in self.word_list:
            assert any(letter == self.center_letter for letter in word), "All words must contain the center letter"
        assert self.max_guesses > 0, "Max guesses must be a positive integer"


class SpellingBeeEnv(NYTGameEnv):

    def __init__(self, config: SpellingBeeConfig, render_mode: str="human"):
        super().__init__(config)
        self.render_mode = render_mode

        self.num_guesses = 0
        self.total_points = 0
        self.words_guessed = set()
        self.valid_words_guessed = []
        self.info= {'history': []}  # To track the history of guesses and rewards

    def reset(self):
        self.num_guesses = 0
        self.total_points = 0
        self.words_guessed = set()
        self.valid_words_guessed = []
        self.info = {'history': []}  # To track the history of guesses and rewards
        return self._get_obs(), self.info  # Return initial observation and info

    def _get_obs(self):
        obs = {
            'num_guesses': self.num_guesses,
            'total_points': self.total_points,
            'words_guessed': list(self.words_guessed)
        }
        return obs

    def step(self, action):
        # TODO: add feedback in message (e.g. too short, missing center letter, etc.)
        self.num_guesses += 1
        was_guessed = action in self.words_guessed
        self.words_guessed.add(action)
        if was_guessed or not self._word_is_valid(action):
            self.info['history'].append((action, 0))
            return self._get_obs(), 0, self._is_terminated(), self._is_truncated(), self.info
        else:
            self.valid_words_guessed.append(action)

        reward = self._get_reward(action)
        self.total_points += reward
        self.info['history'].append((action, reward))
        return self._get_obs(), reward, self._is_terminated(), self._is_truncated(), self.info  # Return a dummy observation, reward, done, and info

    def render(self):
        if self.render_mode == "human":
            print(f"Center letter: {self.config.center_letter} | All letters: {self.config.letter_list}")
            print(f"Total points: {self.total_points} | You have found {len(self.valid_words_guessed)} words: {self.valid_words_guessed}")

    def close(self):
        pass  # No cleanup needed for this environment

    def _get_reward(self, word):
        reward = 0
        if len(word) <4:
            return 0
        elif len(word) == 4:
            return 1
        else:
            reward += len(word)
            if self._is_pangram(word):
                reward += 7
            return reward

    def _is_pangram(self, word):
        return set(word) >= set(self.config.letter_list)

    def _word_is_valid(self, word):
        return (word in self.config.word_list and
                set(word) <= set(self.config.letter_list) and
                self.config.center_letter in word)

    def legal_words(self) -> set[str]:
        """Returns valid puzzle words that have not been guessed yet."""
        return set(self.config.word_list) - set(self.valid_words_guessed)

    def _is_terminated(self) -> bool:
        """
        _is_terminated returns True if all valid words have been guessed,
        and false otherwise.
        """
        return len(self.config.word_list) == len(self.valid_words_guessed)

    def _is_truncated(self) -> bool:
        return self.num_guesses >= self.config.max_guesses


if __name__ == "__main__":
    # Setting up game config
    center_letter = "N"
    letter_list: set[str] = {"T", "A", "F", "M", "O", "R", "N"}
    word_list: set[str] = {"FRONTMAN", "FONT", "AFFRONT", "MAROON", "TORN"}
    max_guesses: int = 10

    config = SpellingBeeConfig(center_letter, letter_list, word_list, max_guesses)

    # Init variables to track game status
    truncated = False   # True if max guesses are used
    terminated = False  # True if all valid words are guessed

    env = SpellingBeeEnv(config)
    while not (truncated or terminated):
        print()
        env.render()

        guess = input("Enter your guess: ").strip().upper()
        obs, reward, terminated, truncated, info = env.step(guess)
        print(f"Reward: {reward}, Total Points: {obs['total_points']}, Num Guesses: {obs['num_guesses']}")
        if truncated or terminated:
            print("Game Over!")
            break
