import google.generativeai as genai
import gymnasium as gym
from dataclasses import dataclass

# ---- Config & Env (imported inline to keep tester self-contained) ---- #

@dataclass
class SpellingBeeConfig:
    center_letter: str
    letter_list: set[str]
    word_list: set[str]
    max_guesses: int = 10

    def validate(self):
        assert self.center_letter in self.letter_list, "Center letter must be in the letter list"
        assert all(letter in self.letter_list for word in self.word_list for letter in word), "All words must be formed from the letter list"
        for word in self.word_list:
            assert any(letter == self.center_letter for letter in word), "All words must contain the center letter"
        assert self.max_guesses > 0, "Max guesses must be a positive integer"


class SpellingBeeEnv(gym.Env):

    def __init__(self, config: SpellingBeeConfig):
        super().__init__()
        self.config = config
        self.num_guesses = 0
        self.total_points = 0
        self.words_guessed = set()
        self.info = {'history': []}

    def reset(self):
        self.num_guesses = 0
        self.total_points = 0
        self.words_guessed = set()
        self.info = {'history': []}
        return self._get_obs(), self.info

    def _get_obs(self):
        return {
            'num_guesses': self.num_guesses,
            'total_points': self.total_points,
            'words_guessed': list(self.words_guessed)
        }

    def step(self, action):
        self.num_guesses += 1
        self.words_guessed.add(action)
        if not self._word_is_valid(action):
            self.info['history'].append((action, 0))
            return self._get_obs(), 0, self._is_done(), self.info
        reward = self._get_reward(action)
        self.total_points += reward
        self.info['history'].append((action, reward))
        return self._get_obs(), reward, self._is_done(), self.info

    def _get_reward(self, word):
        if len(word) < 4:
            return 0
        elif len(word) == 4:
            return 1
        else:
            reward = len(word)
            if self._is_pangram(word):
                reward += 7
            return reward

    def _is_pangram(self, word):
        return set(word) >= set(self.config.letter_list)

    def _word_is_valid(self, word):
        return (word in self.config.word_list and
                set(word) <= set(self.config.letter_list) and
                self.config.center_letter in word)

    def _is_done(self):
        return self.num_guesses >= self.config.max_guesses


# ---- Prompts ---- #

SPELLING_BEE_SYSTEM = """You are an expert NYT Spelling Bee player. In this game you must guess words that:
  1. Are at least 4 letters long.
  2. Use only the allowed letters (letters may be reused).
  3. Always contain the center letter.
A pangram uses every allowed letter at least once and earns a bonus.
Respond with ONLY a single word guess — no explanation, no punctuation."""

SPELLING_BEE_USER = """New Spelling Bee puzzle!
Allowed letters: {letters}
Center letter (must appear in every word): {center}
You have {max_guesses} attempts.
Please guess a word."""


# ---- Main ---- #

def run_test(api_key: str):
    genai.configure(api_key=api_key)

    # Puzzle config
    config = SpellingBeeConfig(
        center_letter="N",
        letter_list={"T", "A", "F", "M", "O", "R", "N"},
        word_list={"FRONTMAN", "FONT", "AFFRONT", "MAROON", "TORN"},
        max_guesses=10,
    )
    config.validate()

    # LLM setup
    model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=SPELLING_BEE_SYSTEM)
    chat = model.start_chat()

    initial_msg = SPELLING_BEE_USER.format(
        letters=", ".join(sorted(config.letter_list)),
        center=config.center_letter,
        max_guesses=config.max_guesses,
    )

    env = SpellingBeeEnv(config)
    obs, info = env.reset()

    # First guess
    response = chat.send_message(initial_msg)
    word = response.text.strip().upper()

    while True:
        obs, reward, done, info = env.step(word)
        print(f"Guess: {word:15s} | Reward: {reward} | Total: {obs['total_points']} | Guesses: {obs['num_guesses']}")

        if done:
            break

        # Build feedback
        if reward > 0:
            fb = f"'{word}' was correct! You earned {reward} point(s). Running total: {obs['total_points']}."
        else:
            fb = f"'{word}' was not accepted. No points earned."
        guessed = ", ".join(obs["words_guessed"]) if obs["words_guessed"] else "none"
        fb += f"\nGuesses used: {obs['num_guesses']}. Words so far: {guessed}.\nPlease guess another word."

        response = chat.send_message(fb)
        word = response.text.strip().upper()

    print(f"\nGame Over! Final score: {obs['total_points']} in {obs['num_guesses']} guesses")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_spelling_bee.py <GEMINI_API_KEY>")
        sys.exit(1)
    run_test(sys.argv[1])
