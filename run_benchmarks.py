"""
Unified benchmark script for NYT Games.

Loads a model once and evaluates it across any combination of games.

Usage (HF local model):
    python run_benchmarks.py --model_path ./model --num_puzzles 100
    python run_benchmarks.py --model_path ./model --games wordle connections --num_puzzles 50
    python run_benchmarks.py --model_path ./model --games all --use_thinking_format

Usage (cloud API):
    python run_benchmarks.py --backend cloud --cloud_model api-gpt-oss-120b --num_puzzles 50
    python run_benchmarks.py --backend cloud --cloud_model my-model --cloud_base_url https://my-api.example.com --api_key "abcdefg"
"""

import argparse
import json
import os
from pathlib import Path

ALL_GAMES = ["wordle", "spelling_bee", "connections", "strands"]


def _load_hf_model(model_path: str):
    """Load a local HF model (merged or LoRA adapter) and return (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

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


def build_backend(args):
    """Construct and return the appropriate GuessBackend from parsed CLI args."""
    from nytgames.benchmarks.backend import HFBackend, CloudBackend

    if args.backend == "cloud":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("Cloud backend requires --api_key or OPENAI_API_KEY env var.")
        return CloudBackend(
            model=args.cloud_model,
            api_key=api_key,
            base_url=args.cloud_base_url,
            use_thinking_format=args.use_thinking_format,
        )
    else:
        model, tokenizer = _load_hf_model(args.model_path)
        return HFBackend(model, tokenizer, use_thinking_format=args.use_thinking_format)


def run_game(game: str, backend, num_puzzles, temperature, verbose, output_dir: Path):
    if game == "wordle":
        from nytgames.benchmarks.wordle_benchmark import WordleBenchmark
        benchmark = WordleBenchmark(backend)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "spelling_bee":
        from nytgames.benchmarks.spelling_bee_benchmark import SpellingBeeBenchmark
        benchmark = SpellingBeeBenchmark(backend)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "connections":
        from nytgames.benchmarks.connections_benchmark import ConnectionsBenchmark
        benchmark = ConnectionsBenchmark(backend)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    elif game == "strands":
        from nytgames.benchmarks.strands_benchmark import StrandsBenchmark
        benchmark = StrandsBenchmark(backend)
        results = benchmark.run(num_puzzles=num_puzzles, temperature=temperature, verbose=verbose)

    else:
        raise ValueError(f"Unknown game: {game}")

    results.print_summary()
    save_path = output_dir / f"{game}_results.json"
    results.save(str(save_path))
    return results


def compute_overall_scores(model_summaries: dict) -> dict:
    """
    Compute normalized overall scores across models and games.

    For each game g, normalize each model's avg_score by the best model on that game:
        s̃_{m,g} = s_{m,g} / max_{m'} s_{m',g}
    Then average across games played by each model:
        Overall_m = (1 / |G_m|) * sum_{g in G_m} s̃_{m,g}

    Args:
        model_summaries: {model_name: {game: {"avg_score": float, ...}}}

    Returns:
        {model_name: overall_score}
    """
    # Collect max avg_score per game across all models
    max_score_per_game = {}
    for game_scores in model_summaries.values():
        for game, metrics in game_scores.items():
            s = metrics.get("avg_score", 0.0)
            if game not in max_score_per_game or s > max_score_per_game[game]:
                max_score_per_game[game] = s

    overall = {}
    for model, game_scores in model_summaries.items():
        normalized = []
        for game, metrics in game_scores.items():
            max_s = max_score_per_game.get(game, 1.0)
            s = metrics.get("avg_score", 0.0)
            normalized.append(s / max_s if max_s > 0 else 0.0)
        overall[model] = sum(normalized) / len(normalized) if normalized else 0.0
    return overall


def print_combined_summary(game_results: dict, output_dir: Path = None, model_name: str = None):
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
        # Save this model's summary as {model_name}_summary.json
        name = model_name or "model"
        model_summary_path = output_dir / f"{name}_summary.json"
        with open(model_summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Model summary saved to {model_summary_path}")

        # Load all saved model summaries and compute overall normalized scores
        all_summaries = {}
        for path in sorted(output_dir.glob("*_summary.json")):
            m = path.stem.removesuffix("_summary")
            with open(path) as f:
                all_summaries[m] = json.load(f)

        if len(all_summaries) >= 1:
            overall_scores = compute_overall_scores(all_summaries)
            print("\n" + "=" * 60)
            print("Overall Normalized Scores  (s̃ = avg_score / max_model_score per game)")
            print("=" * 60)
            print(f"{'Model':<30} {'Overall':>10}")
            print("-" * 60)
            for m, score in sorted(overall_scores.items(), key=lambda x: -x[1]):
                marker = " ←" if m == name else ""
                print(f"{m:<30} {score:>10.4f}{marker}")
            print("=" * 60)

            combined_path = output_dir / "overall_scores.json"
            with open(combined_path, "w") as f:
                json.dump(overall_scores, f, indent=2)
            print(f"Overall scores saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark a model across NYT games.")

    # Backend selection
    parser.add_argument(
        "--backend", choices=["hf", "cloud"], default="hf",
        help="Backend: 'hf' for local HuggingFace model, 'cloud' for OpenAI-compatible API.",
    )

    # HF-specific
    parser.add_argument(
        "--model_path", default="",
        help="Path to local HF model or LoRA adapter directory (required for --backend hf).",
    )

    # Cloud-specific
    parser.add_argument(
        "--cloud_model", default="",
        help="Model name for cloud backend (e.g. 'api-gpt-oss-120b').",
    )
    parser.add_argument(
        "--cloud_base_url", default="https://tritonai-api.ucsd.edu",
        help="Base URL for the OpenAI-compatible API (default: https://tritonai-api.ucsd.edu).",
    )
    parser.add_argument(
        "--api_key", default="",
        help="API key for cloud backend (or set OPENAI_API_KEY env var).",
    )

    # Thinking format
    parser.add_argument(
        "--use_thinking_format", action="store_true", default=False,
        help="Append thinking-format instruction to system prompt and parse 'Answer:' from responses.",
    )

    # Shared
    parser.add_argument(
        "--games", nargs="+", default=["all"],
        help=f"Games to benchmark. Use 'all' or any subset of: {ALL_GAMES}",
    )
    parser.add_argument("--num_puzzles", type=int, default=None, help="Number of puzzles per game (default: all).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0 greedy).")
    parser.add_argument("--output_dir", default="benchmark_results", help="Directory to save result JSONs.")
    parser.add_argument("--verbose", action="store_true", help="Print progress every 50 puzzles.")
    parser.add_argument(
        "--model_name", default="",
        help="Short name used to tag saved results (defaults to model_path basename or cloud_model).",
    )

    args = parser.parse_args()

    # Validate backend-specific required args
    if args.backend == "hf" and not args.model_path:
        parser.error("--model_path is required when --backend hf")
    if args.backend == "cloud" and not args.cloud_model:
        parser.error("--cloud_model is required when --backend cloud")

    games = ALL_GAMES if "all" in args.games else args.games
    for g in games:
        if g not in ALL_GAMES:
            parser.error(f"Unknown game '{g}'. Choose from: {ALL_GAMES}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a model name for tagging saved results
    model_name = args.model_name or (
        Path(args.model_path).name if args.model_path else args.cloud_model
    )

    backend = build_backend(args)

    game_results = {}
    for game in games:
        print(f"\n{'='*60}")
        print(f"Running: {game}")
        print(f"{'='*60}")
        game_results[game] = run_game(
            game=game,
            backend=backend,
            num_puzzles=args.num_puzzles,
            temperature=args.temperature,
            verbose=args.verbose,
            output_dir=output_dir,
        )

    if game_results:
        print_combined_summary(game_results, output_dir=output_dir, model_name=model_name)


if __name__ == "__main__":
    main()
