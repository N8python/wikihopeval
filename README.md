# wikihopeval

Brief evaluator for Wikipedia path-finding trajectories.

**(GPT-5.3-Codex generated)**

## What this repo needs to run

- `main.py`
- `wiki_graph_final.jsonl`
- `sample_trajectory_results_eval.jsonl`

## Run an evaluation

Start your OpenAI-compatible server first (for example LM Studio on `http://localhost:8080/v1`), then run:

```bash
python main.py \
  --graph-jsonl wiki_graph_final.jsonl \
  --eval-jsonl sample_trajectory_results_eval.jsonl \
  --model default_model \
  --base-url http://localhost:8080/v1 \
  --api-key sk-lm-local \
  --parallelism 1 \
  --run-prefix rnj_1_instruct
```

Run again with a different served model and a different prefix (example):

```bash
python main.py --parallelism 1 --model default_model --run-prefix step240
```

Outputs:
- `<run_prefix>_evaluation_results.jsonl` (per-example details, including full conversation trajectory)
- `<run_prefix>_evaluation_summary.json` (aggregate metrics)

If `--run-prefix` is omitted, default names are `evaluation_results.jsonl` and `evaluation_summary.json`.

## How grading works

For each example `i`:
- `L_i` = ground-truth shortest path length from dataset
- `M_i` = model-submitted path length

Per-example score:
- `0` if no submission or invalid path
- `L_i / M_i` if submission is valid

Weighted overall score:

```text
overall_weighted_score = sum(score_i * L_i) / sum(L_i)
```

This weights longer-path questions more heavily.

`evaluation_summary.json` also reports:
- aggregate submission rate
- aggregate accuracy (valid + correct endpoint path)
- per-path-length accuracy
- per-path-length average score
