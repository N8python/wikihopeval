#!/usr/bin/env python3
"""Evaluate a model on sampled Wikipedia pathfinding trajectories."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class EvalExample:
    sample_id: int
    source: str
    target: str
    path_length: int


def build_prompt(start_article: str, target_article: str) -> str:
    return f"""Your task is to find a path from one Wikipedia article to another.

Your starting article is: <article>{start_article}</article>
Your target article is: <article>{target_article}</article>

Available tools:
- get_article: Gets all articles linked from a given article. Args: article_name (str)
- submit_solution: Submits a path from the starting article to the target article. Args: path (list of str)

<format_rules>
- Every response must begin with [think]
- End your thinking with [/think]
- After [/think], include your tool call
- No text outside of [think]...[/think] and <tool_call>...</tool_call>
</format_rules>

Tool format:
<tool_call>
<name>tool_name</name>
<param name="arg_name">value</param>
</tool_call>

Tool Call Example for getting articles linked from "Baroque_architecture":
<tool_call>
<name>get_article</name>
<param name="article_name">Baroque_architecture</param>
</tool_call>

Tool Call Example for submitting a solution path from "Chemical_formula" to "Metabolism":
<tool_call>
<name>submit_solution</name>
<param name="path">["Chemical_formula", "Ethanol", "Metabolism"]</param>
</tool_call>

Do not omit the last node in the path. The last node must be the target article "{target_article}".

Begin!
"""


def load_wiki_graph(graph_jsonl: Path) -> dict[str, list[str]]:
    wiki_dict: dict[str, list[str]] = {}
    with graph_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            wiki_dict[record["vertex"]] = record.get("outgoing_edges", [])
    return wiki_dict


def load_eval_examples(eval_jsonl: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                EvalExample(
                    sample_id=int(record["sample_id"]),
                    source=record["source"],
                    target=record["target"],
                    path_length=int(record["path_length"]),
                )
            )
    return examples


def extract_tool_call(response: str) -> tuple[str | None, dict[str, str], str | None]:
    match = TOOL_CALL_RE.search(response)
    if not match:
        return None, {}, "No <tool_call> block found."

    tool_call_str = match.group(0)
    try:
        tool_call_xml = ET.fromstring(tool_call_str)
    except ET.ParseError as exc:
        return None, {}, f"Malformed tool-call XML: {exc}"

    name_elem = tool_call_xml.find("name")
    if name_elem is None or not name_elem.text:
        return None, {}, "Missing tool name."

    params: dict[str, str] = {}
    for param in tool_call_xml.findall("param"):
        pname = param.attrib.get("name")
        if pname:
            params[pname] = param.text or ""
    return name_elem.text.strip(), params, None


def parse_solution_path(path_str: str) -> list[str] | None:
    try:
        parsed = json.loads(path_str)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(path_str)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(x, str) for x in parsed):
        return None
    return parsed


def validate_path(
    path: list[str],
    start_article: str,
    target_article: str,
    wiki_dict: dict[str, list[str]],
) -> tuple[bool, str]:
    if not path:
        return False, "Empty path."
    if path[0] != start_article:
        return False, f"Path does not start with {start_article}."
    if path[-1] != target_article:
        return False, f"Path does not end with {target_article}."

    for i in range(len(path) - 1):
        cur_article = path[i]
        next_article = path[i + 1]
        if next_article not in wiki_dict.get(cur_article, []):
            return False, f"{next_article} is not linked from {cur_article}."

    return True, "ok"


def run_example(
    client: OpenAI,
    model: str,
    wiki_dict: dict[str, list[str]],
    source: str,
    target: str,
    max_turns: int,
    temperature: float,
) -> dict[str, Any]:
    conversation = [{"role": "user", "content": build_prompt(source, target)}]
    submitted_path: list[str] | None = None
    submit_attempted = False
    last_error = ""

    for _ in range(max_turns):
        completion = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
        )
        response = completion.choices[0].message.content or ""
        conversation.append({"role": "assistant", "content": response})

        tool_name, params, parse_error = extract_tool_call(response)
        if parse_error:
            last_error = parse_error
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"{parse_error} Please include a valid tool call "
                        "in the required XML format."
                    ),
                }
            )
            continue

        if tool_name == "get_article":
            article_name = params.get("article_name", "")
            linked_articles = wiki_dict.get(article_name, [])
            tool_response = (
                f"Articles linked from {article_name}: "
                f"{json.dumps(linked_articles, ensure_ascii=False)}"
            )
            conversation.append({"role": "user", "content": tool_response})
            continue

        if tool_name == "submit_solution":
            submit_attempted = True
            parsed_path = parse_solution_path(params.get("path", ""))
            if parsed_path is None:
                last_error = "Invalid path format in submit_solution."
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Path must be a JSON array of article names, e.g. "
                            '["A", "B", "C"]. Please submit again.'
                        ),
                    }
                )
                continue

            submitted_path = parsed_path
            break

        last_error = f"Unknown tool: {tool_name}"
        conversation.append(
            {"role": "user", "content": f"Unknown tool: {tool_name}. Use a valid tool."}
        )

    if submitted_path is None:
        return {
            "submitted": submit_attempted,
            "valid": False,
            "validation_error": last_error or "No valid submitted path.",
            "model_path_length": None,
            "model_path": None,
            "conversation": conversation,
        }

    valid, validation_error = validate_path(submitted_path, source, target, wiki_dict)
    model_path_length = len(submitted_path) - 1 if valid else None
    return {
        "submitted": True,
        "valid": valid,
        "validation_error": validation_error,
        "model_path_length": model_path_length,
        "model_path": submitted_path,
        "conversation": conversation,
    }


def get_thread_local_client(base_url: str, api_key: str) -> OpenAI:
    key = (base_url, api_key)
    client = getattr(_THREAD_LOCAL, "client", None)
    client_key = getattr(_THREAD_LOCAL, "client_key", None)
    if client is None or client_key != key:
        client = OpenAI(base_url=base_url, api_key=api_key)
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.client_key = key
    return client


def build_result_row(example: EvalExample, run_result: dict[str, Any]) -> dict[str, Any]:
    if run_result["valid"]:
        model_len = run_result["model_path_length"]
        score = example.path_length / model_len if model_len else 0.0
    else:
        score = 0.0

    return {
        "sample_id": example.sample_id,
        "source": example.source,
        "target": example.target,
        "example_path_length": example.path_length,
        "weight": example.path_length,
        "submitted": run_result["submitted"],
        "correct": run_result["valid"],
        "model_path_length": run_result["model_path_length"],
        "score": score,
        "validation_error": run_result["validation_error"],
        "model_path": run_result["model_path"],
        "conversation": run_result["conversation"],
    }


def evaluate(
    model: str,
    base_url: str,
    api_key: str,
    wiki_dict: dict[str, list[str]],
    examples: list[EvalExample],
    max_turns: int,
    temperature: float,
    parallelism: int,
) -> list[dict[str, Any]]:
    if parallelism < 1:
        raise ValueError("--parallelism must be >= 1")

    total = len(examples)
    if total == 0:
        return []

    def run_single(example: EvalExample) -> dict[str, Any]:
        client = get_thread_local_client(base_url=base_url, api_key=api_key)
        run_result = run_example(
            client=client,
            model=model,
            wiki_dict=wiki_dict,
            source=example.source,
            target=example.target,
            max_turns=max_turns,
            temperature=temperature,
        )
        return build_result_row(example, run_result)

    if parallelism == 1:
        results: list[dict[str, Any]] = []
        pbar = tqdm(examples, total=total, desc="Evaluating", unit="example")
        for example in pbar:
            result = run_single(example)
            results.append(result)
            pbar.set_postfix(
                sample_id=example.sample_id,
                length=example.path_length,
                correct=result["correct"],
                score=f"{result['score']:.4f}",
            )
        pbar.close()
        return results

    ordered_results: list[dict[str, Any] | None] = [None] * total
    next_index = 0
    pending: dict[Future[dict[str, Any]], int] = {}

    def submit_one(executor: ThreadPoolExecutor, index: int) -> None:
        future = executor.submit(run_single, examples[index])
        pending[future] = index

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        initial = min(parallelism, total)
        for _ in range(initial):
            submit_one(executor, next_index)
            next_index += 1

        pbar = tqdm(total=total, desc="Evaluating", unit="example")
        while pending:
            done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                result = future.result()
                ordered_results[index] = result

                ex = examples[index]
                pbar.update(1)
                pbar.set_postfix(
                    sample_id=ex.sample_id,
                    length=ex.path_length,
                    correct=result["correct"],
                    score=f"{result['score']:.4f}",
                )

                if next_index < total:
                    submit_one(executor, next_index)
                    next_index += 1
        pbar.close()

    return [r for r in ordered_results if r is not None]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    total_submitted = sum(1 for r in results if r["submitted"])

    weighted_numer = sum(r["score"] * r["weight"] for r in results)
    weighted_denom = sum(r["weight"] for r in results)
    overall_weighted_score = weighted_numer / weighted_denom if weighted_denom else 0.0

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_length[int(r["example_path_length"])].append(r)

    per_length_accuracy_percent: dict[str, float] = {}
    per_length_avg_score: dict[str, float] = {}
    per_length_counts: dict[str, int] = {}
    for length in sorted(by_length):
        rows = by_length[length]
        n = len(rows)
        per_length_counts[str(length)] = n
        per_length_accuracy_percent[str(length)] = (
            100.0 * sum(1 for r in rows if r["correct"]) / n
        )
        per_length_avg_score[str(length)] = sum(r["score"] for r in rows) / n

    return {
        "num_examples": total,
        "submission_rate_percent": 100.0 * total_submitted / total,
        "aggregate_accuracy_percent": 100.0 * total_correct / total,
        "overall_weighted_score": overall_weighted_score,
        "per_path_length_counts": per_length_counts,
        "per_path_length_accuracy_percent": per_length_accuracy_percent,
        "per_path_length_average_score": per_length_avg_score,
        "weighted_score_formula": (
            "sum(score_i * path_length_i) / sum(path_length_i), "
            "where score_i = 0 if incorrect else example_length/model_length"
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def with_run_prefix(path: Path, run_prefix: str | None) -> Path:
    if not run_prefix:
        return path
    return path.with_name(f"{run_prefix}_{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-jsonl",
        type=Path,
        default=Path("wiki_graph_final.jsonl"),
        help="Graph JSONL file used by the get_article tool.",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=Path("sample_trajectory_results_eval.jsonl"),
        help="Evaluation trajectories JSONL.",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("evaluation_results.jsonl"),
        help="Per-example model outputs and scores.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("evaluation_summary.json"),
        help="Aggregate metrics summary file.",
    )
    parser.add_argument(
        "--model",
        default="default_model",
        help="Model name for chat.completions.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "sk-lm-local"),
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Optional prefix for output files. When set, outputs become "
            "<run-prefix>_<filename>."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help=(
            "Number of evaluation examples to run concurrently. "
            "Uses a rolling worker pool."
        ),
    )
    args = parser.parse_args()

    wiki_dict = load_wiki_graph(args.graph_jsonl)
    examples = load_eval_examples(args.eval_jsonl)
    results_path = with_run_prefix(args.results_jsonl, args.run_prefix)
    summary_path = with_run_prefix(args.summary_json, args.run_prefix)
    print(
        f"[load] graph_vertices={len(wiki_dict)} eval_examples={len(examples)}",
        flush=True,
    )
    print("Parrallelism:", args.parallelism, flush=True)

    results = evaluate(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        wiki_dict=wiki_dict,
        examples=examples,
        max_turns=args.max_turns,
        temperature=args.temperature,
        parallelism=args.parallelism,
    )
    summary = summarize(results)

    write_jsonl(results_path, results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(
        f"[done] results_jsonl={results_path} summary_json={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
