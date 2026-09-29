from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from .data import (
    filter_pairs_to_items,
    group_positive_items,
    load_benchmark_queries,
    load_items,
    load_train_pairs,
    split_query_groups,
    validate_dataset_schema,
)
from .engine import SearchEngine
from .evaluation import evaluate_cached_ranker_table, save_json
from .features import FeatureBuilder
from .pipeline import (
    _build_tuning_validation_tables,
    build_item_indexes,
    build_term_expansion,
    ensure_dirs,
    history_artifact_exists,
    item_index_exists,
    load_cached_truths,
    load_retriever,
    make_encoder,
    save_predictions,
    training_signature,
    validate_answer_file,
)
from .ranker import LearningToRankModel, load_ranker_table
from .runtime import configure_runtime, print_runtime


def cmd_check(cfg: Config) -> None:
    runtime = configure_runtime(cfg.raw["compute"]["device"], cfg.raw["compute"]["cpu_threads"])
    print_runtime(runtime)
    print(f"project_root: {cfg.project_root}")
    print(f"data_dir:     {cfg.data_dir}")
    print(f"artifacts:    {cfg.artifact_root}")
    for path in [cfg.train_path, cfg.items_path, cfg.queries_path]:
        print(("OK     " if path.is_file() else "MISSING ") + str(path))
    missing = [p for p in [cfg.train_path, cfg.items_path, cfg.queries_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Dataset files are missing")
    schema = validate_dataset_schema(cfg.train_path, cfg.items_path, cfg.queries_path)
    print(f"train columns:   {len(schema['train_columns'])}")
    print(f"items columns:   {len(schema['items_columns'])}")
    print(f"query columns:   {len(schema['queries_columns'])}")
    print(f"item indexes:    {'READY' if item_index_exists(cfg) else 'NOT BUILT'}")
    print(f"history_train:   {'READY' if history_artifact_exists(cfg, 'history_train') else 'NOT BUILT'}")
    print(f"history_all:     {'READY' if history_artifact_exists(cfg, 'history_all') else 'NOT BUILT'}")


def cmd_build_indexes(cfg: Config) -> None:
    if item_index_exists(cfg):
        print(f"Item indexes are up to date: {cfg.artifact_root}")
        return
    build_item_indexes(cfg)


def _split_data(cfg: Config):
    train_pairs = load_train_pairs(cfg.train_path)
    items = load_items(cfg.items_path)
    train_pairs = filter_pairs_to_items(train_pairs, set(items.item_id.astype(str)))
    grouped = group_positive_items(train_pairs)
    train_groups, val_groups = split_query_groups(
        grouped,
        fit_fraction=float(cfg.raw["training"]["train_fraction"]),
        seed=int(cfg.raw["training"]["split_seed"]),
    )
    return train_pairs, grouped, train_groups, val_groups


def _ranker_artifact_is_current(cfg: Config, name: str = "ranker_training.json") -> bool:
    path = cfg.artifact_root / "reports" / name
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report.get("training_signature") == training_signature(cfg)
    except Exception:
        return False


def _save_validation_from_cache(cfg: Config, ranker: LearningToRankModel, train_groups: pd.DataFrame, val_groups: pd.DataFrame) -> dict:
    signature = training_signature(cfg)
    val_cache = cfg.artifact_root / "cache" / "ranker_validation_table.parquet"
    truth_path = cfg.artifact_root / "cache" / "ranker_validation_truth.json"
    table = load_ranker_table(val_cache, signature)
    if table is None:
        raise RuntimeError("Validation feature cache is missing or stale; run train-ranker first")
    if not truth_path.exists():
        keys = set(table.features["__query_key"].astype(str))
        truth_groups = val_groups[val_groups.query_key.astype(str).isin(keys)].reset_index(drop=True)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(
            json.dumps({str(r.query_key): [str(x) for x in r.positive_item_ids] for r in truth_groups.itertuples(index=False)}, ensure_ascii=False),
            encoding="utf-8",
        )
    truths = load_cached_truths(truth_path)
    report = evaluate_cached_ranker_table(
        table,
        truths,
        ranker,
        cfg.raw,
        profiles=list(cfg.raw["candidate_pool"]["protection_profiles"].keys()),
    )
    save_json(report, cfg.artifact_root / "reports" / "validation.json")
    save_json(
        {"protection_profile": report["selected_profile"], "training_signature": signature},
        cfg.artifact_root / "reports" / "selected_runtime.json",
    )
    return report


def cmd_train_ranker(cfg: Config) -> None:
    ensure_dirs(cfg)
    if not item_index_exists(cfg):
        raise RuntimeError("Item retrieval indexes are missing. Run `uv run avito-retrieval build-indexes` first.")

    train_pairs, _, train_groups, val_groups = _split_data(cfg)
    items = load_items(cfg.items_path)
    feature_builder = FeatureBuilder(items, fuzzy=bool(cfg.raw["features"]["fuzzy"]))
    print(
        f"TRAIN/VAL split by normalized query text: train={len(train_groups):,} query-contexts, "
        f"validation={len(val_groups):,} query-contexts"
    )
    print("Existing item/history artifacts are preserved and reused; no item index rebuild is triggered by training settings.")

    encoder, train_table, val_table, val_eval_queries, signature = _build_tuning_validation_tables(
        cfg, train_groups, val_groups, train_pairs, items, feature_builder
    )

    tuned = LearningToRankModel()
    training_cfg = cfg.raw["ranker"]
    tune_report = tuned.fit_with_validation(
        train_table.features,
        train_table.labels,
        train_table.groups,
        val_table.features,
        val_table.labels,
        val_table.groups,
        training_cfg,
    )
    tuned.save(cfg.artifact_root / "models" / "ranker_tuned.joblib")

    best_iteration = int(tune_report["best_iteration"])
    production = LearningToRankModel()
    production_report = production.fit_full(
        train_table.features,
        train_table.labels,
        train_table.groups,
        training_cfg,
        best_iteration,
    )
    production.save(cfg.artifact_root / "models" / "ranker.joblib")

    # Keep the expensive label-derived artifacts if they already exist; only build the all-train
    # production memory once after the ranker has been tuned.
    all_grouped = group_positive_items(train_pairs)
    all_history_ready = history_artifact_exists(cfg, "history_all", expected_query_count=len(all_grouped))
    all_term_ready = (cfg.artifact_root / "term_expansion.json").exists()
    if not all_history_ready or not all_term_ready:
        from .pipeline import build_history_artifact
        item_map = {item_id: i for i, item_id in enumerate(items.item_id.astype(str))}
        build_history_artifact(cfg, all_grouped, encoder, item_map, "history_all")
        build_term_expansion(cfg, train_pairs, "term_expansion")
    else:
        print("Reusing existing production history/term-expansion artifacts")

    validation_report = _save_validation_from_cache(cfg, tuned, train_groups, val_groups)
    report = {
        "training_signature": signature,
        "split": {
            "strategy": "query_text_disjoint",
            "train_fraction": float(cfg.raw["training"]["train_fraction"]),
            "train_query_contexts": int(len(train_groups)),
            "validation_query_contexts": int(len(val_groups)),
            "validation_ranker_queries_evaluated": int(len(val_eval_queries)),
            "seed": int(cfg.raw["training"]["split_seed"]),
        },
        "ranker": {
            "objective": tune_report["objective"],
            "loss": tune_report["loss_description"],
            "monitor": tune_report["monitor_metric"],
            "best_iteration": best_iteration,
            "best_validation_ndcg_at_50": tune_report["best_validation_ndcg_at_50"],
            "early_stopping_rounds": tune_report["early_stopping_rounds"],
        },
        "training_table": {
            "rows": int(len(train_table.features)),
            "queries": int(len(train_table.groups)),
            "positive_rows": int(train_table.labels.sum()),
            "stats": train_table.stats,
            "cache": str((cfg.artifact_root / "cache" / "ranker_train_table.parquet").resolve()),
        },
        "validation_table": {
            "rows": int(len(val_table.features)),
            "queries": int(len(val_table.groups)),
            "positive_rows": int(val_table.labels.sum()),
            "stats": val_table.stats,
            "cache": str((cfg.artifact_root / "cache" / "ranker_validation_table.parquet").resolve()),
        },
        "validation_metrics": validation_report,
        "production_model": production_report,
        "leakage_control": {
            "query_split": "same normalized search_query is kept inside one split",
            "history": "validation uses history built from train split; train/validation current query key is excluded during retrieval",
            "term_expansion": "validation uses expansion fitted on train split",
        },
    }
    save_json(report, cfg.artifact_root / "reports" / "ranker_training.json")
    print(json.dumps(report["ranker"], ensure_ascii=False, indent=2))
    print("=== VALIDATION RECALL ===")
    print(f"RRF Recall@50:   {validation_report['rrf']['recall@50']:.6f}")
    print(f"Final Recall@50: {validation_report['profiles'][validation_report['selected_profile']]['recall@50']:.6f}")
    print(f"Selected profile: {validation_report['selected_profile']}")


def cmd_validate(cfg: Config) -> None:
    _, _, train_groups, val_groups = _split_data(cfg)
    tuned_path = cfg.artifact_root / "models" / "ranker_tuned.joblib"
    if not tuned_path.exists() or not _ranker_artifact_is_current(cfg):
        raise RuntimeError("Validation ranker is missing or stale. Run `uv run avito-retrieval train-ranker` first.")
    ranker = LearningToRankModel.load(tuned_path)
    report = _save_validation_from_cache(cfg, ranker, train_groups, val_groups)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _selected_profile(cfg: Config) -> str:
    path = cfg.artifact_root / "reports" / "selected_runtime.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("training_signature") == training_signature(cfg):
                return str(data["protection_profile"])
        except Exception:
            pass
    return str(cfg.raw["candidate_pool"]["default_protection_profile"])


def cmd_report(cfg: Config) -> None:
    training_path = cfg.artifact_root / "reports" / "ranker_training.json"
    if not training_path.exists():
        raise FileNotFoundError(f"Missing {training_path}")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    ranker = training["ranker"]
    print("=== RANKER TRAINING ===")
    print(f"training_signature: {training.get('training_signature', 'missing')}")
    print(f"split: train={training['split']['train_query_contexts']:,}, val={training['split']['validation_query_contexts']:,}")
    print(f"loss: {ranker['loss']}")
    print(f"objective: {ranker['objective']}")
    print(f"monitor: {ranker['monitor']}")
    print(f"best_iteration: {ranker['best_iteration']}")
    print(f"best_validation_ndcg@50: {ranker['best_validation_ndcg_at_50']:.6f}")
    validation = training.get("validation_metrics", {})
    if validation:
        selected = validation["selected_profile"]
        print("=== HOLDOUT ===")
        print(f"Ranker NDCG@50: {validation.get('ranker_ndcg@50', float('nan')):.6f}")
        print(f"RRF Recall@50:   {validation['rrf']['recall@50']:.6f}")
        print(f"Final Recall@50: {validation['profiles'][selected]['recall@50']:.6f}")
        print(f"Delta:           {validation['profiles'][selected]['recall@50'] - validation['rrf']['recall@50']:+.6f}")
        print(f"Selected profile: {selected}")


def cmd_predict(cfg: Config, output_path: str | Path) -> None:
    items = load_items(cfg.items_path)
    queries = load_benchmark_queries(cfg.queries_path)
    ranker_path = cfg.artifact_root / "models" / "ranker.joblib"
    if cfg.raw["ranker"]["enabled"] and (not ranker_path.exists() or not _ranker_artifact_is_current(cfg)):
        raise RuntimeError("Production ranker is missing or stale. Run `uv run avito-retrieval train-ranker` first.")
    if not history_artifact_exists(cfg, "history_all") or not (cfg.artifact_root / "term_expansion.json").exists():
        raise RuntimeError("Production label artifacts are missing. Run `uv run avito-retrieval train-ranker` first.")
    retriever = load_retriever(cfg, items, history_name="history_all", term_expansion_name="term_expansion")
    fb = FeatureBuilder(items, fuzzy=bool(cfg.raw["features"]["fuzzy"]))
    ranker = LearningToRankModel.load(ranker_path) if cfg.raw["ranker"]["enabled"] else None
    engine = SearchEngine(retriever, fb, ranker, cfg.raw)
    profile = _selected_profile(cfg)
    predictions = engine.predict(queries, protection_profile=profile, desc="BENCHMARK queries")
    answer = save_predictions(queries.query_id.tolist(), predictions, output_path)
    print(f"saved: {Path(output_path).resolve()}")
    print(f"rows: {len(answer):,}; nonempty answers: {(answer['answer'].str.len() > 0).sum():,}")
    print(f"protection profile: {profile}")
    print("Benchmark Recall@50 cannot be computed locally because relevance labels are hidden.")


def cmd_inspect(cfg: Config, query_id: str | None, query_text: str | None, top_n: int) -> None:
    items = load_items(cfg.items_path)
    if query_id:
        queries = load_benchmark_queries(cfg.queries_path)
        match = queries.loc[queries.query_id.astype(str) == str(query_id)]
        if match.empty:
            raise ValueError(f"query_id not found: {query_id}")
        row = match.iloc[0]
    elif query_text:
        row = pd.Series({
            "search_query": query_text,
            "search_location_id": -1,
            "search_is_delivery_search": 0,
            "search_infm_params_text": "",
            "search_category": 114,
        })
    else:
        raise ValueError("Use --query-id or --query")
    retriever = load_retriever(cfg, items, history_name="history_all", term_expansion_name="term_expansion")
    fb = FeatureBuilder(items, fuzzy=True)
    ranker_path = cfg.artifact_root / "models" / "ranker.joblib"
    ranker = LearningToRankModel.load(ranker_path) if ranker_path.exists() else None
    engine = SearchEngine(retriever, fb, ranker, cfg.raw)
    out = engine.inspect(row, top_n=top_n, protection_profile=_selected_profile(cfg))
    print(out.to_string(index=False))


def cmd_run(cfg: Config, output: str) -> None:
    cmd_check(cfg)
    if not item_index_exists(cfg):
        cmd_build_indexes(cfg)

    ranker_current = (not cfg.raw["ranker"]["enabled"]) or _ranker_artifact_is_current(cfg)
    if not ranker_current:
        print("Ranker artifact is missing or stale; training with cached split/history artifacts where available.")
        cmd_train_ranker(cfg)
    elif not (cfg.artifact_root / "reports" / "validation.json").exists():
        cmd_validate(cfg)

    cmd_predict(cfg, output)
    validate_answer_file(cfg, output)


def cmd_warmup_model(cfg: Config) -> None:
    configure_runtime(cfg.raw["compute"]["device"], cfg.raw["compute"]["cpu_threads"])
    encoder = make_encoder(cfg)
    from .text import dense_query_text
    vector = encoder.encode([dense_query_text("проверка модели", "", 0)], desc="Warm-up embedding")
    print(f"embedding shape: {vector.shape}")
    print(f"device: {encoder.device}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="avito-retrieval")
    parser.add_argument("--config", default="configs/mac_m4pro.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("build-indexes")
    sub.add_parser("train-ranker")
    sub.add_parser("validate")
    sub.add_parser("report")
    sub.add_parser("warmup-model")
    predict = sub.add_parser("predict")
    predict.add_argument("--output", default="answer.csv")
    validator = sub.add_parser("validate-answer")
    validator.add_argument("--answer", default="answer.csv")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--query-id")
    inspect.add_argument("--query")
    inspect.add_argument("--top", type=int, default=20)
    run = sub.add_parser("run")
    run.add_argument("--output", default="answer.csv")

    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    commands = {
        "check": lambda: cmd_check(cfg),
        "build-indexes": lambda: cmd_build_indexes(cfg),
        "train-ranker": lambda: cmd_train_ranker(cfg),
        "validate": lambda: cmd_validate(cfg),
        "report": lambda: cmd_report(cfg),
        "warmup-model": lambda: cmd_warmup_model(cfg),
        "predict": lambda: cmd_predict(cfg, args.output),
        "validate-answer": lambda: validate_answer_file(cfg, args.answer),
        "inspect": lambda: cmd_inspect(cfg, args.query_id, args.query, args.top),
        "run": lambda: cmd_run(cfg, args.output),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
