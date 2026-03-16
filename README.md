# cse291-nytgames

This project contains code and notebooks for using online RL techniques to train language models to play multi-turn lexical reasoning games from the New York Times (NYT) daily puzzle games.

The repo currently supports training on and benchmarking local Huggingface models and cloud-based LLMs on 4 NYT games:
- Wordle
- Spelling Bee
- Strands
- Connections

## Setup

1. To use the `nytgames` module, create and activate the conda environment (or install packages with `pip install -r requirements.txt`)

```
conda env create -f environment.yml
conda activate nytgames
```

For GRPO training requirements, refer to `requirements_grpo.txt`.

## Project Structure

```text
# Directories and files with unused draft notebooks/scripts excluded in tree
├── README.md
├── environment.yml                         # conda env file
├── grpo_training                           # training notebooks for approaches
│   ├── train-grpo-fixed-multiturn.ipynb
│   ├── train_grpo.ipynb
│   └── train_grpo_wordle.ipynb
├── nytgames                                # module containing all testbed and benchmark files
│   ├── README.md
│   ├── __init__.py
│   ├── benchmarks                          # scripts for running model on testbed
│   │   ├── __init__.py
│   │   ├── backend.py                      # HFBackend and CloudBackend implementations
│   │   ├── connections_benchmark.py
│   │   ├── spelling_bee_benchmark.py
│   │   ├── strands_benchmark.py
│   │   └── wordle_benchmark.py
│   ├── data                                # data and scraper files for env config
│   │   ├── SOURCES.md
│   │   ├── __init__.py
│   │   ├── connections.csv
│   │   ├── dataset.py                      # all dataset/dataloader classes for games
│   │   ├── dictionary.txt
│   │   ├── scrapers
│   │   │   └── strands_datascraper.py
│   │   ├── spelling_bee.csv
│   │   ├── strands.csv
│   │   ├── wordle_past_solutions.txt
│   │   └── wordle_valid_guesses.csv
│   ├── env                                 # game classes and configs for RL
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── connections.py
│   │   ├── spellingbee.py
│   │   ├── strands.py
│   │   └── wordle.py
│   ├── prompts                             # game-specific llm prompts
│   │   ├── connections_system.md
│   │   ├── connections_user.md
│   │   ├── spelling_bee_system.md
│   │   ├── spelling_bee_user.md
│   │   ├── strands_system.md
│   │   ├── strands_user.md
│   │   ├── wordle_system.md
│   │   └── wordle_user.md
├── requirements.txt                        # for pip install for nytgames module
├── requirements_grpo.txt                   # for pip install for grpo training
└── run_benchmarks.py                       # cli script for running model against benchmarks
```

## Benchmark Results

You can find the notebooks containing the training code for each of the three approaches below in the `grpo_training` directory:

* Approach 1: Simultaneous Unified Training - `single_simultaneous_grpo.ipynb`
* Approach 2: Sequential Unified Training - `train_grpo_single_model.ipynb`
* Approach 3: Separate Per-Game Adapters - `train_grpo.ipynb`

Overall score is a **relative** metric — each model's per-game score is normalized against the best model on that game, then averaged across games:

$$\tilde{s}_{m,g} = \frac{s_{m,g}}{\max_{m'} s_{m',g}}, \qquad \mathrm{Overall}_m = \frac{1}{|G_m|}\sum_{g \in G_m} \tilde{s}_{m,g}$$

A score of 1.0 means best-in-class on every game played; scores are only meaningful when compared within the same set of models.

The results below reflect post-training results from the first 100 puzzles of the benchmark set.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model / Approach</th>
      <th colspan="2">Wordle</th>
      <th colspan="2">Spelling Bee</th>
      <th colspan="2">Strands</th>
      <th colspan="2">Conections</th>
      <th rowspan="2">Overall Score</th>
    </tr>
    <tr>
      <th># Solved</th><th>Avg Score</th>
      <th>Avg words</th><th>Avg Score</th>
      <th>Spanagrams Found</th><th>Avg Words</th>
      <th># Solved</th><th>Avg Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Approach 1: Simultaneous Unified</td>
      <td>0</td><td>5.2</td>
      <td>2.6</td><td>26.0</td>
      <td>0</td><td>0.4</td>
      <td>0</td><td>0.0</td>
      <td>0.750</td>
    </tr>
    <tr>
      <td>Approach 2: Sequential Unified</td>
      <td>0</td><td>3.4</td>
      <td>1.9</td><td>2.1</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>0.367</td>
    </tr>
    <tr>
      <td>Approach 3: Separate Per-Game Adapters</td>
      <td>0</td><td>4.5</td>
      <td>5.4</td><td>3.3</td>
      <td>0</td><td>0.1</td>
      <td>0</td><td>0.0</td>
      <td>0.311 </td>
    </tr>
    <tr>
      <td>Base Model (Qwen2.5-1.5B)</td>
      <td>0</td><td>0.0</td>
      <td>0.1</td><td>0.2</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>0.004</td>
    </tr>
  </tbody>
</table>




## How to run your model on the NYTGames benchmark

Our puzzle data was scraped between February to March 2026, so any runs on the benchmark set since then are outdated.

See the `run_benchmarks.py` script to run inference on the benchmark set either for a locally stored HuggingFace-compatible model or a cloud-based model through your CLI.

For local models, you will need either the full merged model or the LoRA adapter for inference.

```bash
# Local HF model — all games, 100 puzzles
python run_benchmarks.py --model_path ./model --games all --num_puzzles 100

# Local HF model — specific games
python run_benchmarks.py --model_path ./model --games wordle connections --num_puzzles 50

# Cloud/API model
python run_benchmarks.py --backend cloud --cloud_model gpt-4o --api_key YOUR_KEY --num_puzzles 50

# With thinking-format parsing (e.g. for chain-of-thought models)
python run_benchmarks.py --model_path ./model --use_thinking_format --num_puzzles 100

# Tag and compare multiple models (run once per model; overall_scores.json accumulates)
python run_benchmarks.py --model_path ./model_v1 --model_name baseline --output_dir results/
python run_benchmarks.py --model_path ./model_v2 --model_name finetuned --output_dir results/
```


