from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SUPPORTED_GAMES = {"wordle", "spelling_bee", "strands", "connections"}


def _load_prompt(filename):
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


class LLMHandler:
    """
    Use this AFTER training to play games with the fine-tuned mode
    Training pipeline:  train_grpo.ipynb -> saves LoRA weights into folder /grpo_output
    Evaluation pipeline: LLMHandler loads those weights -> plays games

    Usage:
        # if have fine-tuned local model (after GRPO training):
        handler = LLMHandler(
            game="wordle",
            model_path="path/to/grpo_output",
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            use_lora=True,
        )

        # if want to just use HuggingFace hub model:
        handler = LLMHandler(
            game="wordle",
            model_path="Qwen/Qwen2.5-1.5B-Instruct",
            use_lora=False,
        )
        and keep base_model the same as it will be ignored 
    """

    def __init__(
        self,
        game="wordle",
        model_path = "Qwen/Qwen2.5-1.5B-Instruct",  # change to path of trained model grpo_output folder
        base_model="Qwen/Qwen2.5-1.5B-Instruct",    # only needed if use_lora=True
        use_lora=False,
        temperature=0.7,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        assert game in SUPPORTED_GAMES, f"Game must be one of {SUPPORTED_GAMES}"
        self.game = game
        self.temperature = temperature
        self.device = device
        self.messages = []
        self.config = None

        # Load tokenizer
        tokenizer_path = base_model if (use_lora and base_model) else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if use_lora and base_model:
            # Load fine-tuned LoRA adapter on top of base model
            print(f"Loading base model: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                dtype=torch.float16
                # device_map="auto", # <--- feel free to uncomment
            )
            base = base.to(self.device)
            print(f"Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.to(self.device)
        else:
            # Load da model directly 
            print(f"Loading model: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16
                # device_map="auto",
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model ready on {device} for game: {game}")

    def reset(self, config, **kwargs):
        self.config = config
        self.messages = []
        system_prompt = _load_prompt(f"{self.game}_system.md")

        if self.game == "wordle":
            user_prompt = _load_prompt("wordle_user.md").format(
                max_guesses=config.max_guesses
            )
        elif self.game == "spelling_bee":
            user_prompt = _load_prompt("spelling_bee_user.md").format(
                letters=", ".join(sorted(config.letter_set)),
                center=config.center_letter,
                max_guesses=config.max_guesses,
            )
        elif self.game == "strands":
            user_prompt = _load_prompt("strands_user.md").format(
                theme=config.theme,
                num_theme_words=len(config.theme_words),
                max_guesses=config.max_guesses,
                board_str=kwargs.get("board_str", ""),
            )
        elif self.game == "connections":
            initial_obs = kwargs.get("initial_obs", {})
            words_display = ", ".join(initial_obs.get("remaining_words", []))
            user_prompt = _load_prompt("connections_user.md").format(
                words=words_display,
                max_mistakes=config.max_mistakes,
            )

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_action(self) -> str:
        """Generate the model's next action given current message history."""
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        self.messages.append({"role": "assistant", "content": response.upper()})
        return response.upper()


    # Per game Feedback --------------------------------------------------

    def give_feedback(self, word, reward, obs, **kwargs):
        if self.game == "wordle":
            self._wordle_feedback(word, reward, obs, **kwargs)
        elif self.game == "spelling_bee":
            self._spelling_bee_feedback(word, reward, obs)
        elif self.game == "strands":
            self._strands_feedback(word, reward, obs)
        elif self.game == "connections":
            self._connections_feedback(word, reward, obs)


    def _wordle_feedback(self, word, reward, obs, **kwargs):
        # map env tile states to display labels
        label = {"CORRECT": "GREEN", "PRESENT": "YELLOW", "ABSENT": "GRAY"}

        guess_tiles = next(
            (tiles for w, tiles in obs["guesses"] if w == word), []
        )
        
        letter_feedback = []
        greens = []
        yellows = []
        grays = []
        for i, (letter, tile) in enumerate(zip(word, guess_tiles)):
            color = label[tile]
            letter_feedback.append(f"{letter}({color})")
            if color == "GREEN":
                greens.append(f"{letter} in position {i+1}")
            elif color == "YELLOW":
                yellows.append(f"{letter} is in the word but not position {i+1}")
            else:
                grays.append(letter)

        guess_num = obs["num_guesses"]
        max_guesses = self.config.max_guesses
        solved = obs["solved"]

        if solved:
            msg = (
                f"Correct! '{word}' is the answer. Solved in {guess_num}/{max_guesses} guesses.\n"
                f"Letter feedback: {' | '.join(letter_feedback)}"
            )
        else:
            msg = f"Guess {guess_num}/{max_guesses}: {' | '.join(letter_feedback)}\n"

            if greens:
                msg += f"Confirmed positions: {', '.join(greens)}.\n"
            if yellows:
                msg += f"Right letters, wrong position: {', '.join(yellows)}.\n"
            if grays:
                msg += f"Not in word: {', '.join(grays)}.\n"

            msg += (
                f"Attempts remaining: {max_guesses - guess_num}.\n"
                f"Use this information to make a better guess. "
                f"Reply with exactly one 5-letter word."
            )

        self.messages.append({"role": "user", "content": msg})

    def _spelling_bee_feedback(self, word, reward, obs):
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
            f"Please guess another word using only the allowed letters, "
            f"making sure it contains the center letter."
        )
        self.messages.append({"role": "user", "content": msg})

    def _strands_feedback(self, word, reward, obs):
        feedback = obs.get("feedback", "")
        if reward == 5:
            msg = f"'{word}': {feedback} Spangram! +5 points."
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