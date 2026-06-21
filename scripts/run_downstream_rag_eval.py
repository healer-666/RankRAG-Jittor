"""Run downstream RAG answer generation for selected rerankers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from src.rag_answer_generation import (
        ContextBuilder,
        RankingsLoader,
        TransformersGenerator,
        evaluate_generated_row,
        load_config,
        load_existing_results,
        read_jsonl,
        write_jsonl_append,
    )
    from src.utils import ensure_parent, resolve_path, write_json
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.rag_answer_generation import (
        ContextBuilder,
        RankingsLoader,
        TransformersGenerator,
        evaluate_generated_row,
        load_config,
        load_existing_results,
        read_jsonl,
        write_jsonl_append,
    )
    from src.utils import ensure_parent, resolve_path, write_json


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(resolve_path(".")))
    except ValueError:
        return path.name


def parse_methods(value: str | None, config: dict) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(config["methods"])


def run_method(
    *,
    method: str,
    context_rows: list[dict],
    generator: TransformersGenerator,
    output_dir: Path,
) -> dict:
    top_k = int(context_rows[0]["top_k"]) if context_rows else 3
    output_path = output_dir / f"{method}_top{top_k}_answers.jsonl"
    existing = load_existing_results(output_path)
    generated = skipped = 0
    start = time.time()
    for context_row in context_rows:
        key = (method, context_row["query_id"], int(context_row["top_k"]))
        if key in existing:
            skipped += 1
            continue
        answer, runtime = generator.generate(context_row["prompt"])
        result = evaluate_generated_row(context_row, answer, runtime)
        write_jsonl_append(output_path, result)
        generated += 1
    return {
        "method": method,
        "output_path": display_path(output_path),
        "generated": generated,
        "skipped_existing": skipped,
        "runtime_sec": time.time() - start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--generator-model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    methods = parse_methods(args.methods, config)
    top_k = int(args.top_k or config["top_k"])
    if args.generator_model_path:
        config["runtime_generator_model_path"] = args.generator_model_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_rows = read_jsonl(config["dataset_path"])
    if args.max_questions is not None:
        qa_rows = qa_rows[: args.max_questions]
    prompt_style = config.get("prompt_style", "original")
    builder = ContextBuilder(
        qa_rows,
        top_k,
        int(config.get("max_context_tokens", 2048)),
        prompt_style,
    )
    loader = RankingsLoader(builder.candidate_pool)
    method_contexts = {}
    for method in methods:
        spec = config["rankings"][method]
        rankings = loader.load_rankings(method, spec["path"], spec.get("nested_key"))
        method_contexts[method] = builder.build(method, rankings)

    generator = TransformersGenerator(config)
    run_summary = {
        "config": args.config,
        "methods": methods,
        "top_k": top_k,
        "generator_model": config["generator_model"],
        "generator_model_path": None,
        "local_model_path_used": bool(generator.used_local_path),
        "prompt_style": builder.prompt_style,
        "prompt_version": config.get("prompt_version", builder.prompt_version),
        "generator_load_time_sec": generator.load_time_sec,
        "device": str(generator.device),
        "device_name": generator.device_name,
        "dtype": generator.dtype,
        "do_sample": config.get("do_sample"),
        "temperature": config.get("temperature"),
        "max_new_tokens": config.get("max_new_tokens"),
        "max_context_tokens": config.get("max_context_tokens"),
        "outputs": [],
    }
    for method in methods:
        run_summary["outputs"].append(
            run_method(
                method=method,
                context_rows=method_contexts[method],
                generator=generator,
                output_dir=output_dir,
            )
        )
    write_json(output_dir / "generation_run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
