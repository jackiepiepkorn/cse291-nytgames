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
# modified prompts to include dictionary and clarify rules for better LLM performance
SPELLING_BEE_SYSTEM = """You are playing a strict word game.
Rules:
- Output EXACTLY one uppercase word.
- Word MUST be in the provided dictionary list.
- Word MUST be at least 4 letters.
- Word MUST contain the center letter.
- Do NOT invent new words.
- Do NOT output explanations, punctuation, or multiple words.
If unsure, repeat a previous valid word."""

SPELLING_BEE_USER = """Allowed letters: {letters}
Center letter: {center}
Valid dictionary words: {dictionary}
You have {max_guesses} attempts.
Output one word."""


# ---- Main ---- #

def run_test(api_key: str):
    genai.configure(api_key=api_key)

    # Puzzle config
    # modified to include a full example with a known solution for better testing
    # guesses are limited to 5 for now to force the LLM to find a valid word quickly
    config = SpellingBeeConfig(
        center_letter="H",
        letter_list={"G", "I", "P", "R", "A", "C", "H"},
        word_list={"GRAPHIC","HIGHCHAIR","PARAGRAPH", "ARCHAIC", "CHICHI", "PARIAH", "AARGH", "CHAIR", "CHICA", "CHIRP", "GRAPH", "PARCH",
                   "ARCH", "CHAI", "CHAP", "CHAR", "CHIA","CHIA", "CHIC", "CHIP", "HAIR", "HARP", "HIGH", "RICH"},
        max_guesses=5,
    )
    config.validate()

    # LLM setup
    # changed from gemini-2.0-flash to gemini-2.5-flash for better performance on the more complex prompt
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SPELLING_BEE_SYSTEM)
    chat = model.start_chat()

    initial_msg = SPELLING_BEE_USER.format(
        letters=", ".join(sorted(config.letter_list)),
        center=config.center_letter,
        dictionary = ", ".join(sorted(config.word_list)),
        max_guesses=config.max_guesses,
    )

    env = SpellingBeeEnv(config)
    obs, info = env.reset()

    # First guess
    response = chat.send_message(initial_msg)
    word = response.text.strip().upper()

    while True:
        print(f"RAW WORD REPR: {repr(word)}")
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
