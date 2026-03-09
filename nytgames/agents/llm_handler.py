from pathlib import Path
from huggingface_hub import InferenceClient
from nytgames.env.wordle import _format_feedback

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SUPPORTED_GAMES = {"wordle", "spelling_bee", "strands", "connections"}

def _load_prompt(filename):
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()

class LLMHandler:
    def __init__(self, game="wordle", model="HuggingFaceH4/zephyr-7b-beta", temperature=0.7):
        self.game = game
        self.client = InferenceClient(model)
        self.model = model
        self.temperature = temperature
        self.messages = []

    def reset(self, config, **kwargs):
        self.config = config
        system_prompt = _load_prompt(f"{self.game}_system.md")
        if self.game == "wordle":
            user_prompt = _load_prompt(f"{self.game}_user.md").format(
                max_guesses=config.max_guesses
            )
        elif self.game == "spelling_bee":
            user_prompt = _load_prompt(f"{self.game}_user.md").format(
                letters=", ".join(sorted(config.letter_set)),
                center=config.center_letter,
                max_guesses=config.max_guesses
            )
        elif self.game == "strands":
            user_prompt = _load_prompt(f"{self.game}_user.md").format(
                theme=config.theme,
                num_theme_words=len(config.theme_words),
                max_guesses=config.max_guesses,
                board_str=kwargs.get("board_str", ""),
            )
        elif self.game == "connections":
            initial_obs = kwargs.get("initial_obs", {})
            words_display = ", ".join(initial_obs.get("remaining_words",[]))
            user_prompt = _load_prompt("connections_user.md").format(
                words=words_display,
                max_mistakes=config.max_mistakes
            )
        self.messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}]

    def get_action(self):
        response = self.client.chat_completion(
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=32,
        )
        word = response.choices[0].message.content.strip().upper()
        self.messages.append({"role": "assistant", "content": word})
        return word

    def give_feedback(self, word, reward, obs, **kwargs):
        if self.game == "wordle":
            self._wordle_feedback(word, reward, obs, **kwargs)
        elif self.game == "spelling_bee":
            self._spelling_bee_feedback(word, reward, obs)
        elif self.game == "strands":
            self._strands_feedback(word, reward, obs)
        elif self.game == "connections":
            self._connections_feedback(word, reward, obs)

    def _wordle_feedback(self, word, reward, obs, guess_num=0, max_guesses=6):
        guess_tiles = next(
            (tiles for w, tiles in obs["guesses"] if w == word), None
        )
        if guess_tiles is not None:
            feedback_str = _format_feedback(word, guess_tiles)
        else:
            feedback_str = obs.get("feedback", f"No feedback available for '{word}'.")

        guess_num = obs["num_guesses"]
        max_guesses = self.config.max_guesses
        solved = obs["solved"]

        if solved:
            msg = (
                f"'{word}' is correct. You solved it on guess {guess_num}/{max_guesses}.\n"
                f"Feedback: {feedback_str}\n"
                f"Reward: {reward:.1f}"
            )
        else:
            msg = (
                f"Guess {guess_num}/{max_guesses}: '{word}'\n"
                f"Feedback: {feedback_str}\n"
                f"Reward: {reward:.1f}\n"
                f"Attempts remaining: {max_guesses - guess_num}.\n"
                f"Please guess another 5-letter word."
            )
        self.messages.append({"role": "user", "content": msg})

    def _spelling_bee_feedback(self, word, reward, obs, max_guesses = 10):
        feedback = obs.get("feedback", "")
        if reward > 0:
            msg = (
                f"'{word}': {feedback} You earned {reward} point(s). "
                f"Running total: {obs['total_points']}."
            )
        else:
            msg = f"'{word}': {feedback} No points earned."

        found = ", ".join(obs["valid_words_guessed"]) if obs["valid_words_guessed"] else "none"
        msg += (
            f"\nGuesses used: {obs['num_guesses']}. "
            f"Words found so far: {found}.\n"
            f"Please guess another word if not all words guessed and have not used {max_guesses} guesses."
        )
        self.messages.append({"role": "user", "content": msg})

    def _strands_feedback(self, word, reward, obs):
        feedback = obs.get("feedback", "")
        if reward == 5:
            msg = f"'{word}': {feedback} Spanagram! +5 points."
        elif reward == 1:
            msg = f"'{word}': {feedback} +1 point."
        else:
            msg = f"'{word}': {feedback}"
        msg += (
            f"\n{obs['progress']}"
            f"\nGuesses used: {obs['num_guesses']}."
            f"\n\nUpdated board:\n{obs['board_str']}"
            f"\nPlease guess another word."
        )
        self.messages.append({"role": "user", "content": msg})


    def _connections_feedback(self, word, reward, obs):
        feedback = obs.get("feedback", "")
        mistakes_left = obs.get("mistakes_remaining", "?")
        
        if reward >= 1:
            solved = obs.get("solved_categories", {})
            last_category = list(solved.keys())[-1]
            msg = (
                f"Correct! Category '{last_category}' found. +{int(reward)} point(s).\n"
                f"Solved so far: {', '.join(solved.keys())}"
            )
        else:
            msg = f"'{feedback}' Mistakes remaining: {mistakes_left}"
        
        remaining = obs.get("remaining_words", [])
        if remaining:
            msg += f"\nRemaining words: {', '.join(remaining)}.\nGuess 4 words separated by commas."
        else:
            msg += "\nPuzzle complete!"
        
        self.messages.append({"role": "user", "content": msg})
