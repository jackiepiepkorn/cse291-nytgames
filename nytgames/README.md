# nytgames

## Running individual envs/games in terminal:

If a environment/game loop is set up in `main`, you can run the python file with `-m` (gnore warnings).

Example: `python -m nytgames.env.spellingbee`

## Online DPO training (Spelling Bee)

Run:

`python -m nytgames.train.online_dpo_spellingbee --model-name Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --pairs-per-round 64`

Useful flags:
- `--csv-path nytgames/data/spelling_bee.csv`
- `--max-puzzles 200`
- `--rollout-steps 1`
- `--train-on-latest-only`
- `--output-dir checkpoints/spellingbee_online_dpo`

Each round prints collection/training/eval stats and writes checkpoints to `output-dir/round_*`.
