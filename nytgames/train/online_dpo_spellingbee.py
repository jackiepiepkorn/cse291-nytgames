"""
Online DPO training for the Spelling Bee environment.

Example:
  python -m nytgames.train.online_dpo_spellingbee \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --rounds 3 \
    --pairs-per-round 64
"""

from __future__ import annotations

import argparse
import inspect
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from nytgames.env.spellingbee import SpellingBeeConfig, SpellingBeeEnv

try:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:
    raise SystemExit(
        "Missing training dependencies. Install: datasets transformers trl accelerate peft"
    ) from exc


WORD_RE = re.compile(r"[A-Z]+")
DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "spelling_bee.csv"
PROMPT_TEMPLATE = """You are playing NYT Spelling Bee.
Allowed letters: {letters}
Center letter: {center}
Guesses used: {num_guesses}/{max_guesses}
Words guessed: {guessed_words}
Respond with exactly one uppercase word."""


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: int
    letter_list: set[str]
    center_letter: str
    word_list: set[str]
    max_guesses: int

    def to_config(self) -> SpellingBeeConfig:
        return SpellingBeeConfig(
            center_letter=self.center_letter,
            letter_list=self.letter_list,
            word_list=self.word_list,
            max_guesses=self.max_guesses,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online DPO training on SpellingBeeEnv")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/spellingbee_online_dpo"))
    parser.add_argument("--max-puzzles", type=int, default=200)
    parser.add_argument("--max-guesses", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pairs-per-round", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--epochs-per-round", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-on-latest-only",
        action="store_true",
        help="If set, DPO updates only use the newest round of pairs.",
    )
    return parser.parse_args()


def load_puzzles(csv_path: Path, max_guesses: int, max_puzzles: int | None = None) -> list[Puzzle]:
    df = pd.read_csv(csv_path)
    if max_puzzles is not None:
        df = df.head(max_puzzles)

    puzzles: list[Puzzle] = []
    for row in df.itertuples(index=False):
        letters = set(str(row.letters).strip().upper())
        center = str(row.center).strip().upper()
        word_list = {
            word.strip().upper()
            for word in str(row.solutions).split(";")
            if word and word.strip()
        }
        puzzles.append(
            Puzzle(
                puzzle_id=int(row.puzzle_id),
                letter_list=letters,
                center_letter=center,
                word_list=word_list,
                max_guesses=max_guesses,
            )
        )
    return puzzles


def clone_env(env: SpellingBeeEnv) -> SpellingBeeEnv:
    cloned = SpellingBeeEnv(env.config, render_mode=env.render_mode)
    cloned.num_guesses = env.num_guesses
    cloned.total_points = env.total_points
    cloned.words_guessed = set(env.words_guessed)
    cloned.valid_words_guessed = list(env.valid_words_guessed)
    history = env.info.get("history", [])
    cloned.info = {"history": list(history)}
    return cloned


def build_prompt(env: SpellingBeeEnv) -> str:
    obs = env._get_obs()
    guessed_words = sorted(obs["words_guessed"])
    guessed_str = ", ".join(guessed_words) if guessed_words else "none"
    return PROMPT_TEMPLATE.format(
        letters=", ".join(sorted(env.config.letter_list)),
        center=env.config.center_letter,
        num_guesses=obs["num_guesses"],
        max_guesses=env.config.max_guesses,
        guessed_words=guessed_str,
    )


def extract_word(text: str) -> str:
    match = WORD_RE.search(text.upper())
    return match.group(0) if match else ""


def sample_action(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = encoded.to(model.device)
    do_sample = temperature > 0

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=do_sample,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    start = encoded["input_ids"].shape[-1]
    completion_tokens = generated[0, start:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    word = extract_word(completion_text)
    return word if word else "ZZZZ"


def estimate_return(env: SpellingBeeEnv, first_action: str, rollout_steps: int, rng: random.Random) -> float:
    sim_env = clone_env(env)
    _, reward, terminated, truncated, _ = sim_env.step(first_action)
    total_return = float(reward)

    for _ in range(rollout_steps):
        if terminated or truncated:
            break
        legal = sorted(sim_env.legal_words())
        if not legal:
            break
        followup = rng.choice(legal)
        _, reward, terminated, truncated, _ = sim_env.step(followup)
        total_return += float(reward)
    return total_return


def choose_pair(
    env: SpellingBeeEnv,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[dict[str, str], dict[str, Any]]:
    prompt = build_prompt(env)
    a = sample_action(
        model,
        tokenizer,
        prompt,
        args.max_prompt_tokens,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    b = sample_action(
        model,
        tokenizer,
        prompt,
        args.max_prompt_tokens,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    if a == b:
        b = sample_action(
            model,
            tokenizer,
            prompt,
            args.max_prompt_tokens,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )

    score_a = estimate_return(env, a, args.rollout_steps, rng)
    score_b = estimate_return(env, b, args.rollout_steps, rng)

    if score_a > score_b:
        chosen, rejected = a, b
    elif score_b > score_a:
        chosen, rejected = b, a
    else:
        # Break ties randomly to keep learning signal stochastic.
        chosen, rejected = (a, b) if rng.random() < 0.5 else (b, a)

    _, reward, terminated, truncated, _ = env.step(chosen)
    pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    metrics = {
        "chosen_reward": float(reward),
        "chosen_return": max(score_a, score_b),
        "terminated": terminated,
        "truncated": truncated,
    }
    return pair, metrics


def build_dpo_trainer(
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> DPOTrainer:
    cfg_candidates = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation,
        "num_train_epochs": args.epochs_per_round,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "max_prompt_length": args.max_prompt_tokens,
        "max_length": args.max_prompt_tokens + args.max_new_tokens + 16,
        "remove_unused_columns": False,
    }
    cfg_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    cfg_kwargs = {k: v for k, v in cfg_candidates.items() if k in cfg_params}
    cfg = DPOConfig(**cfg_kwargs)
    common = {
        "model": model,
        "ref_model": ref_model,
        "args": cfg,
        "train_dataset": train_dataset,
    }
    trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
    if "max_prompt_length" in trainer_params:
        common["max_prompt_length"] = args.max_prompt_tokens
    if "max_length" in trainer_params:
        common["max_length"] = args.max_prompt_tokens + args.max_new_tokens + 16

    # TRL changed the argument name from tokenizer -> processing_class.
    try:
        return DPOTrainer(tokenizer=tokenizer, **common)
    except TypeError:
        return DPOTrainer(processing_class=tokenizer, **common)


def load_model_for_training(model_name: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def evaluate_policy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    puzzles: list[Puzzle],
    eval_episodes: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    solved = 0
    scores: list[float] = []

    for _ in range(eval_episodes):
        puzzle = rng.choice(puzzles)
        env = SpellingBeeEnv(puzzle.to_config())
        obs, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            prompt = build_prompt(env)
            guess = sample_action(
                model,
                tokenizer,
                prompt,
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            obs, _, terminated, truncated, _ = env.step(guess)

        if terminated:
            solved += 1
        scores.append(float(obs["total_points"]))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    solve_rate = solved / eval_episodes if eval_episodes > 0 else 0.0
    return {"avg_score": avg_score, "solve_rate": solve_rate}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    puzzles = load_puzzles(args.csv_path, max_guesses=args.max_guesses, max_puzzles=args.max_puzzles)
    if not puzzles:
        raise SystemExit("No puzzles loaded. Check --csv-path and --max-puzzles.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise SystemExit("Tokenizer has no pad/eos/unk token. Please choose a different model.")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_model_for_training(args.model_name, dtype=dtype)
    ref_model = load_model_for_training(args.model_name, dtype=dtype)

    if torch.cuda.is_available():
        model = model.to("cuda")
        ref_model = ref_model.to("cuda")
    model.train()
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_buffer: list[dict[str, str]] = []

    current_env: SpellingBeeEnv | None = None
    for round_idx in range(args.rounds):
        round_pairs: list[dict[str, str]] = []
        round_rewards: list[float] = []
        round_returns: list[float] = []
        episodes_finished = 0

        while len(round_pairs) < args.pairs_per_round:
            if current_env is None:
                puzzle = rng.choice(puzzles)
                current_env = SpellingBeeEnv(puzzle.to_config())
                current_env.reset()

            pair, metrics = choose_pair(current_env, model, tokenizer, args, rng)
            round_pairs.append(pair)
            round_rewards.append(metrics["chosen_reward"])
            round_returns.append(metrics["chosen_return"])

            if metrics["terminated"] or metrics["truncated"]:
                episodes_finished += 1
                current_env = None

        replay_buffer.extend(round_pairs)
        if len(replay_buffer) > args.replay_buffer_size:
            replay_buffer = replay_buffer[-args.replay_buffer_size:]

        training_pairs = round_pairs if args.train_on_latest_only else replay_buffer
        train_dataset = Dataset.from_list(training_pairs)
        round_dir = args.output_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        trainer = build_dpo_trainer(model, ref_model, tokenizer, train_dataset, round_dir, args)
        trainer.train()
        model = trainer.model
        model.train()

        eval_metrics = evaluate_policy(
            model,
            tokenizer,
            puzzles,
            eval_episodes=args.eval_episodes,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + round_idx + 1,
        )
        trainer.save_model(str(round_dir))
        tokenizer.save_pretrained(str(round_dir))

        avg_reward = sum(round_rewards) / len(round_rewards)
        avg_return = sum(round_returns) / len(round_returns)
        print(
            f"[round {round_idx + 1}/{args.rounds}] "
            f"pairs={len(round_pairs)} replay={len(replay_buffer)} "
            f"episodes={episodes_finished} "
            f"avg_chosen_reward={avg_reward:.3f} "
            f"avg_chosen_return={avg_return:.3f} "
            f"eval_avg_score={eval_metrics['avg_score']:.3f} "
            f"eval_solve_rate={eval_metrics['solve_rate']:.3f}"
        )

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Training complete. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
