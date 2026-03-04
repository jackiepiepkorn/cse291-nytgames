import random
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from ..env.spellingbee import SpellingBeeConfig

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "spelling_bee_system.md").read_text().strip()
_USER_PROMPT_TEMPLATE = (_PROMPTS_DIR / "spelling_bee_user.md").read_text().strip()

_SPELLING_BEE_CSV_PATH = Path(__file__).parent / "spelling_bee.csv"
_DICTIONARY_TXT_PATH = Path(__file__).parent / "dictionary.txt"

def load_dictionary(txt_path: Path = _DICTIONARY_TXT_PATH, length: int = None) -> set[str]:
    if length is not None and length <= 0:
        raise ValueError(f"Length {length} must be > 0 or None for all words.")

    dictionary = [word.strip().upper() for word in open(txt_path).read().split()]
    if length is None:
        return set(dictionary)
    else:
        return set([word for word in dictionary if len(word) == length])

class SpellingBeeDataset(Dataset):
    """
    One item per puzzle row in spelling_bee.csv.

    Each item is a dict with:
        "prompt"    - list of chat messages (system + user) ready for a tokenizer's
                      apply_chat_template(); compatible with TRL GRPOTrainer
        "puzzle_id" - int puzzle identifier (useful for seeding the env in reward fn)
    """

    def __init__(self, max_guesses: int = 10, csv_path: Path = _SPELLING_BEE_CSV_PATH):
        df = pd.read_csv(csv_path)
        df["letters"] = df["letters"].apply(lambda s: set(str(s).upper()))
        df["center"] = df["center"].str.upper()
        df["solutions"] = df["solutions"].apply(
            lambda s: set(w.upper() for w in s.split(";"))
        )
        self._rows = df[["puzzle_id", "letters", "center", "solutions"]].to_dict(
            orient="records"
        )
        self._max_guesses = max_guesses

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        user_msg = _USER_PROMPT_TEMPLATE.format(
            letters=", ".join(sorted(row["letters"])),
            center=row["center"],
            max_guesses=self._max_guesses,
        )
        return {
            "prompt": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "puzzle_id": int(row["puzzle_id"]),
        }

    def get_config(self, puzzle_id: int, max_guesses: int = None) -> SpellingBeeConfig:
        """Return a SpellingBeeConfig for the given puzzle_id."""
        max_guesses = max_guesses if max_guesses is not None else self._max_guesses
        try:
            row = next(r for r in self._rows if r["puzzle_id"] == puzzle_id)
        except StopIteration:
            raise ValueError(f"puzzle_id {puzzle_id} not found in dataset")
        return SpellingBeeConfig(
            center_letter=row["center"],
            letter_set=row["letters"],
            word_set=row["solutions"],
            max_guesses=max_guesses,
        )

    def sample(self, max_guesses: int = None) -> tuple[SpellingBeeConfig, int]:
        """Return a (SpellingBeeConfig, puzzle_id) for a random puzzle."""
        row = random.choice(self._rows)
        config = self.get_config(row["puzzle_id"], max_guesses=max_guesses)
        return config, int(row["puzzle_id"])
