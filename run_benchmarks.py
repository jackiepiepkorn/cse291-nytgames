"""
Unified benchmark script for NYT Games.

Loads a model once and evaluates it across any combination of games.

Usage:
    python run_benchmarks.py --model_path ./model --num_puzzles 100  # runs all games
    python run_benchmarks.py --model_path ./model --games wordle connections --num_puzzles 50
    python run_benchmarks.py --model_path ./model --games all --output_dir ./results
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ALL_GAMES = ["wordle", "spelling_bee", "connections", "strands"]


def load_model(model_path: str):
    model_path = Path(model_path)
    adapter_config_path = model_path / "adapter_config.json"

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", adapter_cfg.get("base_model"))
        print(f"Loading base model: {base_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, tokenizer


def run_game(game: str, model_path: str, model, tokenizer, num_puzzles, temperature, verbose, output_dir: Path):
    if game == "wordle":
        from nytgames.benchmarks.wordle_benchmark import WordleBenchmark
        benchmark = WordleBenchmark(model_path, model=model, tokenizer=tokenizer)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "spelling_bee":
        from nytgames.benchmarks.spelling_bee_benchmark import SpellingBeeBenchmark
        benchmark = SpellingBeeBenchmark(model_path, model=model, tokenizer=tokenizer)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "connections":
        from nytgames.benchmarks.connections_benchmark import ConnectionsBenchmark
        benchmark = ConnectionsBenchmark(model_path, model=model, tokenizer=tokenizer)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "strands":
        from nytgames.benchmarks.strands_benchmark import StrandsBenchmark
        benchmark = StrandsBenchmark(model_path, model=model, tokenizer=tokenizer)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    else:
        raise ValueError(f"Unknown game: {game}")

    results.print_summary()
    save_path = output_dir / f"{game}_results.json"
    results.save(str(save_path))
    return results


def print_combined_summary(game_results: dict, output_dir: Path = None):
    print("\n" + "=" * 60)
    print("Combined Benchmark Summary")
    print("=" * 60)
    print(f"{'Game':<20} {'Solve Rate':>12} {'Avg Score':>12}")
    print("-" * 60)
    summary = {}
    for game, results in game_results.items():
        solve_rate = f"{results.solve_rate:.1%}"
        avg_score = f"{results.avg_score:.1f}"
        print(f"{game:<20} {solve_rate:>12} {avg_score:>12}")
        summary[game] = {"solve_rate": results.solve_rate, "avg_score": results.avg_score}
    print("=" * 60)

    if output_dir is not None:
        save_path = output_dir / "summary.json"
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a model across NYT games.")
    parser.add_argument("--model_path", required=True, help="Path to the model or LoRA adapter directory.")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["all"],
        help=f"Games to benchmark. Use 'all' or any subset of: {ALL_GAMES}",
    )
    parser.add_argument("--num_puzzles", type=int, default=None, help="Number of puzzles per game (default: all).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0 greedy).")
    parser.add_argument("--output_dir", default="benchmark_results", help="Directory to save result JSONs.")
    parser.add_argument("--verbose", action="store_true", help="Print progress every 50 puzzles.")
    args = parser.parse_args()

    games = ALL_GAMES if "all" in args.games else args.games
    for g in games:
        if g not in ALL_GAMES:
            parser.error(f"Unknown game '{g}'. Choose from: {ALL_GAMES}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)

    game_results = {}
    for game in games:
        print(f"\n{'='*60}")
        print(f"Running: {game}")
        print(f"{'='*60}")
        game_results[game] = run_game(
            game=game,
            model_path=args.model_path,
            model=model,
            tokenizer=tokenizer,
            num_puzzles=args.num_puzzles,
            temperature=args.temperature,
            verbose=args.verbose,
            output_dir=output_dir,
        )

    if game_results:
        print_combined_summary(game_results, output_dir=output_dir)


if __name__ == "__main__":
    main()
