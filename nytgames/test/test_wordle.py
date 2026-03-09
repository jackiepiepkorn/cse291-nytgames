from nytgames.env.wordle import WordleConfig, WordleEnv, _format_feedback, _score_guess


def _make_env(target_word="CRANE"):
    word_set = {"CRANE", "SLATE", "CARVE", "BRINE"}
    return WordleEnv(WordleConfig(target_word=target_word, word_set=word_set, max_guesses=6))


def test_format_feedback_uses_words_by_default():
    tiles = _score_guess("SLATE", "CRANE")
    assert _format_feedback("SLATE", tiles) == (
        "S=GRAY L=GRAY A=GREEN T=GRAY E=GREEN"
    )


def test_format_feedback_supports_emoji_rendering():
    tiles = _score_guess("SLATE", "CRANE")
    assert _format_feedback("SLATE", tiles, style="emoji") == "⬜⬜🟩⬜🟩  SLATE"


def test_env_feedback_is_word_based_for_valid_guess():
    env = _make_env()
    env.reset()

    obs, reward, terminated, truncated, _ = env.step("SLATE")

    assert reward == 1
    assert not terminated
    assert not truncated
    assert obs["feedback"] == "S=GRAY L=GRAY A=GREEN T=GRAY E=GREEN"


def test_env_feedback_marks_correct_solution():
    env = _make_env()
    env.reset()

    obs, reward, terminated, truncated, _ = env.step("CRANE")

    assert reward == 20
    assert terminated
    assert not truncated
    assert obs["feedback"] == (
        "C=GREEN R=GREEN A=GREEN N=GREEN E=GREEN - Correct!"
    )
