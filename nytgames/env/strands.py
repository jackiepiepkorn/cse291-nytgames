from dataclasses import dataclass
from typing import Set, List, Tuple, Any, Dict
from .base import NYTGameEnv
import random


def _find_all_paths(board: List[List[str]], word: str) -> List[List[Tuple[int, int]]]:
    """DFS search for all connected paths on the board that spell `word`.
    Returns a list of coord paths, each a list of (row, col) tuples."""
    height = len(board)
    width = len(board[0])
    results = []

    def dfs(char_idx, row, col, path, visited):
        if char_idx == len(word):
            results.append(list(path))
            return
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < height and 0 <= nc < width
                    and (nr, nc) not in visited
                    and board[nr][nc] == word[char_idx]):
                path.append((nr, nc))
                visited.add((nr, nc))
                dfs(char_idx + 1, nr, nc, path, visited)
                path.pop()
                visited.remove((nr, nc))

    for r in range(height):
        for c in range(width):
            if board[r][c] == word[0]:
                dfs(1, r, c, [(r, c)], {(r, c)})

    return results


def _resolve_word_paths(
    all_paths: Dict[str, List[List[Tuple[int, int]]]]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Backtracking search for a consistent assignment of paths (no two words share a cell).
    Returns the first valid assignment found.
    """
    # Most-constrained-variable first: try words with fewest paths first to maximize pruning
    words = sorted(all_paths.keys(), key=lambda w: len(all_paths[w]))

    def backtrack(idx, used_coords, assignment):
        if idx == len(words):
            return dict(assignment)
        word = words[idx]
        for path in all_paths[word]:
            path_coords = frozenset(path)
            if not path_coords & used_coords:
                assignment[word] = path
                result = backtrack(idx + 1, used_coords | path_coords, assignment)
                if result is not None:
                    return result
                del assignment[word]
        return None

    result = backtrack(0, frozenset(), {})
    assert result is not None, "No valid path assignment found — words may not tile the board."
    return result


def _word_spans_board(coords: List[Tuple[int, int]], width: int, height: int) -> bool:
    """True if the coords touch both edges of a row or both edges of a column."""
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    return (0 in cols and (width - 1) in cols) or (0 in rows and (height - 1) in rows)


@dataclass
class StrandsConfig:
    """
    Configuration for a Strands puzzle.

    Attributes:
        board:        2D grid of letters, board[row][col], uppercase.
        theme_words:  All theme word strings, including the spanagram.
        spanagram:    The special word that spans the full board (row or column edge-to-edge).
        theme:        A short hint phrase relating all theme words.
        dictionary:   Optional reference word list enabling the hint system. If None, hints
                      are disabled and non-theme guesses give no feedback.
        max_guesses:  Episode length; defaults to 3x the number of theme words.

    On construction, __post_init__ runs DFS across the board to locate each theme word's
    path of adjacent cells, then uses backtracking to find a conflict-free assignment
    (no two words share a cell). This serves as validation: if a word can't be placed,
    or the assignment fails, an assertion is raised. The result is stored in word_lookup
    as a plain dict mapping each word string to its list of (row, col) coords.
    """
    board: List[List[str]]       # board[row][col], uppercase letters
    theme_words: frozenset[str]  # all theme word strings (includes spanagram)
    spanagram: str               # must be in theme_words
    theme: str                   # a short hint phrase relating the words
    dictionary: Set[str] | None = None  # if None, hints are disabled
    max_guesses: int | None = None      # defaults to 3x number of theme words

    def __post_init__(self):
        # Normalize to uppercase
        self.board = [[c.upper() for c in row] for row in self.board]
        self.theme_words = frozenset(w.upper() for w in self.theme_words)
        self.spanagram = self.spanagram.upper()

        # Derive dimensions
        self.height: int = len(self.board)
        self.width: int = len(self.board[0]) if self.board else 0

        # Search for each word's path(s) on the board, then resolve conflicts
        all_paths = {word: _find_all_paths(self.board, word) for word in self.theme_words}
        for word, paths in all_paths.items():
            assert len(paths) > 0, f"Word '{word}' not found on the board."
        resolved = _resolve_word_paths(all_paths)

        # word_lookup: maps each theme word to its list of (row, col) coords
        self.word_lookup: Dict[str, List[Tuple[int, int]]] = resolved

        if self.max_guesses is None:
            self.max_guesses = 3 * len(self.word_lookup)

        self.validate()

    def validate(self):
        """Assert board dimensions, full coverage, no overlapping words, and spanagram constraint."""
        assert self.width > 0 and self.height > 0, "Board must have positive dimensions."
        assert len(self.theme) > 0, "Theme must be a non-empty string."
        assert len(self.theme_words) >= 2, "Puzzle must have at least 2 theme words."
        assert self.spanagram in self.theme_words, "Spanagram must be one of the theme words."

        # No two words share a cell (guaranteed by _resolve_word_paths, but double-check)
        all_coords = [coord for coords in self.word_lookup.values() for coord in coords]
        assert len(all_coords) == len(set(all_coords)), "Two words share a cell."

        # Full board coverage
        expected = self.width * self.height
        assert len(all_coords) == expected, \
            f"Board has {expected} cells but words cover {len(all_coords)}."

        # Spanagram spans the board
        assert _word_spans_board(self.word_lookup[self.spanagram], self.width, self.height), \
            "Spanagram must span the board."


class StrandsEnv(NYTGameEnv):
    """
    Gymnasium-compatible environment for the NYT Strands.

    The player is shown a grid of letters and must identify words that are spelled out
    by adjacent connected cells (8-directional). All theme words together tile the entire
    board with no overlaps. When a theme word is correctly guessed, its cells are removed
    from the visible board. The env validates guesses by looking the word up in the
    pre-computed word_lookup table (built during config construction via DFS).

    The spanagram is a special theme word that spans the full board edge-to-edge and is
    worth more points. If a dictionary is provided, guessing valid non-theme words that
    exist on the board earns hint credits; every 3 such words unlocks a scrambled hint
    revealing the letters of an unguessed theme word.

    Observation (dict):
        board_str:           Text-formatted board with row/col indices, "." for cleared cells.
        theme:               The theme hint phrase.
        progress:            Human-readable summary of words found and (if hints enabled)
                             how many non-theme words until the next hint.
        theme_words_guessed: List of correctly found theme words.
        num_guesses:         Total guesses made so far.
        feedback:            Human-readable result of the last guess, or None on reset.
        (Full attempt history including invalid guesses is in info['history'].)

    Action:
        A string word guess (case-insensitive). The player spells out a word they see
        on the board by tracing adjacent letters; only the string is submitted.

    Reward:
        0: invalid guess (not a theme word, already found, etc.)
        1: valid theme word
        5: valid spanagram

    Termination: all theme words have been found.
    Truncation:  max_guesses reached.
    """
    def __init__(self, config: StrandsConfig, render_mode: str="human"):
        super().__init__(config)
        self.render_mode: str = render_mode

        # Game attributes
        self.num_guesses: int = 0
        self.theme_words_guessed: Set[str] = set()
        self.hints_words_guessed: Set[str] = set()
        self.visible_board: List[List[str]] = self._init_visible_board()

        # Debugging info
        self.info: Dict[str, Any] = {"history": []}

    def _init_visible_board(self) -> List[List[str]]:
        """Return a deep copy of the full board. Cells are blanked as theme words are guessed."""
        if self.num_guesses > 0:
            raise Exception("Should only be called before guesses are made!")
        return [row[:] for row in self.config.board]

    def reset(self):
        self.num_guesses = 0
        self.theme_words_guessed = set()
        self.hints_words_guessed = set()
        self.visible_board = self._init_visible_board()
        self.info = {"history": []}
        return self._get_obs(), self.info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Process a word guess. Valid theme words are removed from the visible board.
        If hints are enabled, non-theme board words accumulate hint credits (every 3 earns
        a scrambled hint). Invalid guesses score 0 with no board change.
        """
        action = action.strip().upper()
        self.num_guesses += 1

        valid, feedback = self._word_is_valid(action)
        if valid:
            self.theme_words_guessed.add(action)
            self._remove_from_board(action)
            reward = self._get_reward(action)
        else:
            reward = 0

            # Check for hint
            valid_hint, feedback_hint = self._hint_word_is_valid(action)
            feedback += " " + feedback_hint
            if self.config.dictionary is not None and valid_hint:
                self.hints_words_guessed.add(action)
                if len(self.hints_words_guessed) % 3 == 0:
                    hint = self._give_hint()
                    if hint:
                        feedback += f" Hint: unscramble {''.join(hint)} for a theme word."

        self.info['history'].append((action, reward, feedback))
        return self._get_obs(), reward, self._is_terminated(), self._is_truncated(), self.info

    def _board_to_str(self) -> str:
        """Render the visible board as an indexed ASCII grid; cleared cells shown as '.'."""
        col_header = "   " + " ".join(str(c) for c in range(self.config.width))
        rows = [col_header]
        for r, row in enumerate(self.visible_board):
            cells = " ".join(c if c else "." for c in row)
            rows.append(f"{r}: {cells}")
        return "\n".join(rows)

    def _get_obs(self) -> Dict[str, Any]:
        return {
            'board_str': self._board_to_str(),
            'theme': self.config.theme,
            'progress': (
                f"{len(self.theme_words_guessed)} out of {len(self.config.theme_words)} theme words guessed."
                + (f" {3 - len(self.hints_words_guessed) % 3} more non-theme words before hint." if self.config.dictionary is not None else "")
            ),
            'theme_words_guessed': list(self.theme_words_guessed),
            'num_guesses': self.num_guesses,
            'feedback': self.info['history'][-1][2] if self.info['history'] else None,
        }

    def render(self):
        """
        If render_mode == "human":
            Remaining letters are in non-guessed words.
            Spaces will replace correctly-guessed words.
        """
        if self.render_mode == "human":
            print(self._board_to_str())
        else:
            raise NotImplementedError("Not yet implemented")

    def close(self):
        pass

    def _is_theme_word(self, word) -> bool:
        return word in self.config.theme_words

    def _is_spanagram(self, word) -> bool:
        return word == self.config.spanagram

    def _remove_from_board(self, word: str) -> None:
        """Blank out the cells of a guessed word on the visible board."""
        for row, col in self.config.word_lookup[word]:
            self.visible_board[row][col] = ""

    def _hint_word_is_valid(self, word) -> Tuple[bool, str]:
        """Check if word is a new, valid dictionary word that exists as a connected path on the board. Returns (valid, feedback)."""
        if self.config.dictionary is None:
            raise ValueError("Hints not enabled-- needs reference dictionary.")

        if word not in self.config.dictionary:
            return False, "Not a word!"
        elif word in self.hints_words_guessed:
            return False, "Already guessed!"
        elif len(_find_all_paths(self.config.board, word)) == 0:
            return False, "Word is not on board!"
        else:
            return True, "Found non-theme word!"

    def _give_hint(self) -> List[str]:
        """Return the letters of a random unguessed non-spanagram theme word, shuffled."""
        unguessed_theme_words = list(self.config.theme_words - self.theme_words_guessed - {self.config.spanagram})
        if not unguessed_theme_words:
            return []  # no non-spanagram words left to hint

        word_to_hint = random.choice(unguessed_theme_words)
        scrambled = list(word_to_hint)
        random.shuffle(scrambled)
        return scrambled

    def _word_is_valid(self, word: str) -> Tuple[bool, str]:
        """Check if word is a new, valid theme word. Returns (valid, feedback)."""
        # Cover invalid cases
        if len(word) < 4:
            return False, "Too short"
        elif word in self.theme_words_guessed:
            return False, "Already found!"
        elif not self._is_theme_word(word):
            return False, "Not a theme word."

        # Valid cases
        if self._is_spanagram(word):
            return True, "Spanagram!"
        return True, "Found a theme word!"

    def _get_reward(self, word: str) -> float:
        """Assumes word is a valid, unguessed theme word. +5 for spanagram, +1 for regular."""
        return 5 if self._is_spanagram(word) else 1

    def _is_terminated(self) -> bool:
        """True when all theme words have been found."""
        return self.theme_words_guessed == self.config.theme_words

    def _is_truncated(self) -> bool:
        """True when max_guesses have been used."""
        return self.num_guesses >= self.config.max_guesses


if __name__ == "__main__":
    # Run with python -m nytgames.env.strands
    from nytgames import load_dictionary

    board = [
        list("COWESS"),
        list("HSODKI"),
        list("TBRPRN"),
        list("OOKLTE"),
        list("ANIOUR"),
        list("PTNEGN"),
        list("YRGSPA"),
        list("IFIWEC"),
    ]
    dictionary = load_dictionary()
    print("dict length:", len(dictionary))

    config = StrandsConfig(
        board=board,
        theme_words=frozenset({"WIFI", "DESKS", "PRINTER", "LOUNGE", "PANTRY", "BOOTHS", "COWORKINGSPACE"}),
        spanagram="COWORKINGSPACE",
        theme="Home office alternative",
        dictionary=dictionary
    )
    print(f"Theme: {config.theme}")
    print(f"Words ({len(config.word_lookup)}): {sorted(config.word_lookup.keys())}")
    print(f"Spanagram: {config.spanagram}")
    print(f"Max guesses: {config.max_guesses}")
    print()

    # Init variables to track game status
    truncated = False   # True if max guesses are used
    terminated = False  # True if all valid words are guessed

    env = StrandsEnv(config)
    while not (truncated or terminated):
        print()
        env.render()

        guess = input("\nEnter your guess: ").strip().upper()
        obs, reward, terminated, truncated, info = env.step(guess)
        print(obs['feedback'])
        print(obs['progress'])
        print(f"Reward: {reward}, Num Guesses: {obs['num_guesses']}, Theme words guessed: {obs['theme_words_guessed']}")
        if truncated or terminated:
            print("Game Over!")
            break
