import random
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from ..env.spellingbee import SpellingBeeConfig
from ..env.strands import StrandsConfig
from ..env.connections import ConnectionsConfig, CATEGORIES_COUNT, WORDS_PER_CATEGORY

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_SPELLING_BEE_CSV_PATH = Path(__file__).parent / "spelling_bee.csv"
_DICTIONARY_TXT_PATH = Path(__file__).parent / "dictionary.txt"
_STRANDS_CSV_PATH = Path(__file__).parent / "strands.csv"
_CONNECTIONS_CSV_PATH = Path(__file__).parent / "connections.csv"

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
    _SYSTEM_PROMPT = (_PROMPTS_DIR / "spelling_bee_system.md").read_text().strip()
    _USER_PROMPT_TEMPLATE = (_PROMPTS_DIR / "spelling_bee_user.md").read_text().strip()

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
        user_msg = self._USER_PROMPT_TEMPLATE.format(
            letters=", ".join(sorted(row["letters"])),
            center=row["center"],
            max_guesses=self._max_guesses,
        )
        return {
            "prompt": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
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

class StrandsDataset(Dataset):
    """
    One item per puzzle row in strands.csv.

    Each item is a dict with:
        "prompt"    - list of chat messages (system + user) ready for a tokenizer's
                      apply_chat_template(); compatible with TRL GRPOTrainer
        "puzzle_id" - int puzzle identifier (useful for seeding the env in reward fn)
    This board is stored as board[row][col] - 8 rows x 6 cols, all uppercase.

    CSV columns used:
        puzzle_id, date, clue, spanagram, theme_words, num_theme_words,
        board_row_0 .. board_row_7, board_flat
    """
    _SYSTEM_PROMPT = (_PROMPTS_DIR / "strands_system.md").read_text().strip()
    _USER_PROMPT_TEMPLATE = (_PROMPTS_DIR / "strands_user.md").read_text().strip()

    def __init__(self, csv_path: Path = _STRANDS_CSV_PATH):
        csv_path = csv_path or self._CSV_PATH
        df = pd.read_csv(csv_path)

        # Drop rows without a complete board
        df = df[df["board_flat"].notna() & (df["board_flat"].str.len() == 48)]
        df = df.reset_index(drop=True)

        df["spanagram"]        = df["spanagram"].str.upper()
        df["clue"]             = df["clue"].fillna("")
        df["theme_words_list"] = df["theme_words"].apply(
            lambda s: [w.strip().upper() for w in str(s).split("|") if w.strip()]
        )

        self._rows = df.to_dict(orient="records")

        # Pre-parse board[row][col] once at load time
        for i, row in enumerate(self._rows):
            cleaned = {k.strip(): v for k, v in row.items()}
            self._rows[i] = cleaned

            cleaned["board"] = [
                list(cleaned[f"board_row_{j}"].upper())
                for j in range(8)
            ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        num_theme = int(row["num_theme_words"])
        user_msg = self._USER_PROMPT_TEMPLATE.format(
            theme=row["clue"],
            board_str=_format_strands_board(row["board"]),
            num_theme_words=num_theme,
            max_guesses=3 * num_theme,
        )
        return {
            "prompt": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "puzzle_id": int(row["puzzle_id"]),
        }

    def get_config(self, puzzle_id: int) -> StrandsConfig:
        try:
            row = next(r for r in self._rows if int(r["puzzle_id"]) == puzzle_id)
        except StopIteration:
            raise ValueError(f"puzzle_id {puzzle_id} not found in StrandsDataset")
        return StrandsConfig(
            board=row["board"],
            spanagram=row["spanagram"],
            theme_words=set(row["theme_words_list"]),
            theme=row["clue"],
        )

    def sample(self) -> tuple[StrandsConfig, int]:
        row = random.choice(self._rows)
        return self.get_config(int(row["puzzle_id"])), int(row["puzzle_id"])


class ConnectionsDataset(Dataset):
    _SYSTEM_PROMPT = (_PROMPTS_DIR / "connections_system.md").read_text().strip()
    _USER_PROMPT_TEMPLATE = (_PROMPTS_DIR / "connections_user.md").read_text().strip()
    
    def __init__(self, csv_path: Path = _CONNECTIONS_CSV_PATH):
        df = pd.read_csv(csv_path)

        # Normalize
        df["Word"] = df["Word"].str.upper()
        df["Group Name"] = df["Group Name"].str.upper()

        # Group by Game ID → build categories dict per puzzle
        puzzles = []
        for game_id, group in df.groupby("Game ID"):
            categories = {}
            for group_name, cat_df in group.groupby("Group Name"):
                categories[group_name] = cat_df["Word"].tolist()

            # Only keep complete puzzles (4 categories × 4 words = 16 words)
            total_words = sum(len(v) for v in categories.values())
            if len(categories) == CATEGORIES_COUNT and total_words == CATEGORIES_COUNT * WORDS_PER_CATEGORY:
                puzzle_date = group["Puzzle Date"].iloc[0]
                puzzles.append({
                    "puzzle_id": int(game_id),
                    "puzzle_date": puzzle_date,
                    "categories": categories,
                })

        self._rows = puzzles

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        # Flatten and shuffle words for the prompt
        all_words = [w for words in row["categories"].values() for w in words]
        random.shuffle(all_words)
        user_msg = self._USER_PROMPT_TEMPLATE.format(
            words=", ".join(all_words),
        )
        return {
            "prompt": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "puzzle_id": int(row["puzzle_id"]),
        }

    def get_config(self, puzzle_id: int) -> ConnectionsConfig:
        try:
            row = next(r for r in self._rows if r["puzzle_id"] == puzzle_id)
        except StopIteration:
            raise ValueError(f"puzzle_id {puzzle_id} not found in ConnectionsDataset")
        return ConnectionsConfig(categories=row["categories"])

    def sample(self) -> tuple[ConnectionsConfig, int]:
        row = random.choice(self._rows)
        return self.get_config(row["puzzle_id"]), int(row["puzzle_id"])

# module level helper
def _format_strands_board(board: list[list[str]]) -> str:
    """Render 8×6 Strands board as a readable grid string for prompts."""
    return "\n".join("  ".join(row) for row in board)
